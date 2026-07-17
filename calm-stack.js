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

    // Enable Spot Capacity
    cluster.addAsgCapacityProvider('SpotProvider', {
      autoScalingGroup: cluster.addCapacity('ASG', {
        instanceType: new ec2.InstanceType('g5.xlarge'),
        spotPrice: '0.40', // Set a reasonable ceiling
      }),
      capacityProviderName: 'FARGATE_SPOT', // Or EC2 Spot logic
    });

    // Persistent EFS File System
    const fileSystem = new efs.FileSystem(this, 'CalmEFS', { vpc });
    const volName = 'PersistentData';
    
    const taskDefinition = new ecs.Ec2TaskDefinition(this, 'CalmTask');
    taskDefinition.addVolume({
      name: volName,
      efsVolumeConfiguration: { fileSystemId: fileSystem.fileSystemId }
    });

    const container = taskDefinition.addContainer('CalmContainer', {
      image: ecs.ContainerImage.fromDockerImageAsset(new ecrAssets.DockerImageAsset(this, 'CalmImage', {
        directory: path.join(__dirname, '..'),
      })),
      memoryLimitMiB: 16384,
    });

    container.addMountPoints({
      containerPath: '/app/data',
      sourceVolume: volName,
      readOnly: false
    });

    new ecs.Ec2Service(this, 'CalmService', {
      cluster,
      taskDefinition,
      capacityProviderStrategies: [{ capacityProvider: 'EC2', weight: 1 }]
    });
  }
}
