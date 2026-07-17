const { Stack, RemovalPolicy } = require('aws-cdk-lib');
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
      machineImage: ecs.EcsOptimizedImage.amazonLinux2023(),
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
        directory: path.join(__dirname, '..'),
      })),
      memoryLimitMiB: 16384,
      portMappings: [{ containerPort: 8000, hostPort: 8000 }],
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
