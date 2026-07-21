const { Stack, RemovalPolicy, Duration } = require('aws-cdk-lib');
const ec2 = require('aws-cdk-lib/aws-ec2');
const ecs = require('aws-cdk-lib/aws-ecs');
const efs = require('aws-cdk-lib/aws-efs');
const ecrAssets = require('aws-cdk-lib/aws-ecr-assets');
const path = require('path');

class CalmStack extends Stack {
  constructor(scope, id, props) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, 'CalmVpc', { maxAzs: 2 });
    const cluster = new ecs.Cluster(this, 'CalmCluster', { vpc });

    // Spot capacity setup
    cluster.addCapacity('SpotCapacity', {
      instanceType: new ec2.InstanceType('g5.xlarge'),
      spotPrice: '0.40',
      minCapacity: 1,
      // FIX: Standard AL2023 lacks NVIDIA drivers. Use the GPU-optimized AMI.
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.GPU),
    });

    const fileSystem = new efs.FileSystem(this, 'CalmEFS', { 
      vpc, 
      removalPolicy: RemovalPolicy.DESTROY 
    });

    const taskDefinition = new ecs.Ec2TaskDefinition(this, 'CalmTask');
    taskDefinition.addVolume({
      name: 'PersistentData',
      efsVolumeConfiguration: { fileSystemId: fileSystem.fileSystemId }
    });

    const container = taskDefinition.addContainer('CalmContainer', {
      image: ecs.ContainerImage.fromDockerImageAsset(new ecrAssets.DockerImageAsset(this, 'CalmImage', {
        directory: path.join(__dirname, '../..'),
      })),
      memoryLimitMiB: 16384,
      gpuCount: 1, // FIX: ECS must be told to allocate the GPU to this specific container
      portMappings: [{ containerPort: 8000, hostPort: 8000 }],
      healthCheck: {
        command: [
          "CMD-SHELL", 
          "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\")' || exit 1"
        ],
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        retries: 3,
        startPeriod: Duration.seconds(90), 
      }
    });

    container.addMountPoints({
      containerPath: '/app/data',
      sourceVolume: 'PersistentData',
      readOnly: false
    });

    // Create a service and explicitly open port 8000
    const service = new ecs.Ec2Service(this, 'CalmService', {
      cluster,
      taskDefinition,
      // FIX: Enable Deployment Circuit Breaker
      circuitBreaker: { rollback: true }
    });

    // Directly allow traffic to the instances on port 8000
    service.connections.securityGroups[0].addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8000),
      'Allow public access to dashboard'
    );
  }
}
module.exports = { CalmStack };