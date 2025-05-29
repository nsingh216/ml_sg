import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader
import boto3

# Setup
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()

# Create S3 paths for checkpoints
checkpoint_s3_uri = f's3://{bucket}/bird-detection/checkpoints'
output_s3_uri = f's3://{bucket}/bird-detection/output'

print(f"Checkpoint URI: {checkpoint_s3_uri}")
print(f"Output URI: {output_s3_uri}")

env = {
    'SAGEMAKER_REQUIREMENTS': 'requirements.txt'
}

# SPOT INSTANCE CONFIGURATION
estimator = PyTorch(
    entry_point="train.py",  # Your optimized training script
    source_dir="source",
    role=role,
    env=env,
    framework_version="2.6.0",
    py_version="py312",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    
    # 💰 SPOT INSTANCE SETTINGS
    use_spot_instances=True,
    max_wait=7200,  # Wait up to 2 hours for spot capacity
    max_run=43200,  # Max 12 hours training time
    checkpoint_s3_uri=checkpoint_s3_uri,  # Where to save/restore checkpoints
    
    # Output location
    output_path=output_s3_uri,
    
    # Hyperparameters for your optimized training
    hyperparameters={
        'epochs': 100,
        'batch-size': 16,
        'use-amp': True,
        'image-size': 512,
        'accumulation-steps': 2,
        'learning-rate': 0.005,
        'log-interval': 50,
        'save-every': 10,  # Save checkpoint every 10 epochs
        'checkpoint-dir': '/opt/ml/checkpoints',  # SageMaker checkpoint directory
    },
    
    # Enable metric reporting
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'Loss: ([0-9\\.]+)'},
        {'Name': 'train:speed', 'Regex': 'Speed: ([0-9\\.]+) batch/s'},
        {'Name': 'epoch:time', 'Regex': 'completed in ([0-9\\.]+) minutes'}
    ]
)

print("🚀 Starting spot instance training...")
print("💰 Expected cost savings: 50-70% vs on-demand")
print("⚠️  Note: Training may be interrupted and resumed automatically")

# Start training
try:
    estimator.fit(wait=True)
    print("✅ Training completed successfully!")
    
    # Get final model location
    model_uri = estimator.model_data
    print(f"📦 Model saved to: {model_uri}")
    
except Exception as e:
    print(f"❌ Training failed or was interrupted: {e}")
    print("💡 Check CloudWatch logs for details")
    print(f"🔄 Checkpoints available at: {checkpoint_s3_uri}")


# COST ANALYSIS AFTER TRAINING
def analyze_training_cost(estimator):
    """Calculate actual vs on-demand costs"""
    try:
        # Get training job details
        training_job_name = estimator.latest_training_job.name
        sm_client = boto3.client('sagemaker')
        
        job_details = sm_client.describe_training_job(TrainingJobName=training_job_name)
        
        billable_seconds = job_details['BillableTimeInSeconds']
        training_seconds = job_details['TrainingTimeInSeconds']
        
        # Cost calculation
        on_demand_rate = 4.284  # USD per hour for ml.p3.2xlarge
        spot_rate = job_details.get('FinalMetricDataList', {}).get('SpotPrice', on_demand_rate * 0.3)  # Estimate if not available
        
        on_demand_cost = (billable_seconds / 3600) * on_demand_rate
        actual_cost = (billable_seconds / 3600) * spot_rate
        savings = on_demand_cost - actual_cost
        savings_percent = (savings / on_demand_cost) * 100
        
        print("\n" + "="*50)
        print("💰 COST ANALYSIS")
        print("="*50)
        print(f"Training time: {training_seconds/60:.1f} minutes")
        print(f"Billable time: {billable_seconds/60:.1f} minutes")
        print(f"On-demand cost: ${on_demand_cost:.2f}")
        print(f"Spot cost: ${actual_cost:.2f}")
        print(f"Savings: ${savings:.2f} ({savings_percent:.1f}%)")
        print("="*50)
        
    except Exception as e:
        print(f"Could not analyze costs: {e}")

# Run cost analysis after training
if 'estimator' in locals():
    analyze_training_cost(estimator)


# RESUMING INTERRUPTED TRAINING
def resume_from_checkpoint():
    """Resume training from the last checkpoint"""
    
    # Create new estimator with same settings
    resume_estimator = PyTorch(
        entry_point="train.py",
        source_dir="source", 
        role=role,
        env=env,
        framework_version="2.6.0",
        py_version="py312",
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        
        # Spot settings
        use_spot_instances=True,
        max_wait=7200,
        max_run=43200,
        checkpoint_s3_uri=checkpoint_s3_uri,  # Same checkpoint location
        
        output_path=output_s3_uri,
        
        # Add resume flag
        hyperparameters={
            'epochs': 100,
            'batch-size': 16,
            'use-amp': True,
            'image-size': 512,
            'accumulation-steps': 2,
            'learning-rate': 0.005,
            'log-interval': 50,
            'save-every': 10,
            'checkpoint-dir': '/opt/ml/checkpoints',
            'resume': True,  # Signal to resume from checkpoint
        }
    )
    
    print("🔄 Resuming training from checkpoint...")
    resume_estimator.fit(wait=True)
    
    return resume_estimator

# Uncomment to resume if training was interrupted:
# resume_estimator = resume_from_checkpoint()


# MONITORING SPOT PRICE (Optional)
def check_current_spot_prices():
    """Check current spot prices for p3.2xlarge"""
    ec2 = boto3.client('ec2')
    
    try:
        response = ec2.describe_spot_price_history(
            InstanceTypes=['p3.2xlarge'],
            ProductDescriptions=['Linux/UNIX'],
            MaxResults=5
        )
        
        print("\n🏷️  Current Spot Prices (p3.2xlarge):")
        for price in response['SpotPriceHistory'][:3]:
            az = price['AvailabilityZone']
            spot_price = float(price['SpotPrice'])
            savings = ((4.284 - spot_price) / 4.284) * 100
            print(f"  {az}: ${spot_price:.3f}/hour ({savings:.0f}% savings)")
            
    except Exception as e:
        print(f"Could not fetch spot prices: {e}")

# Check current prices
check_current_spot_prices()
