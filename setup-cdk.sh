#!/bin/bash

brew install node
npm install -g aws-cdk

PROJECT_DIR="calm-cdk"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" || exit

cdk init app --language javascript
npm install aws-cdk-lib constructs dotenv

cat <<EOF > .env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
EOF

echo ".env" >> .gitignore

echo "Setup complete. Update .env, replace lib/${PROJECT_DIR}-stack.js with your script, then run: cdk bootstrap && cdk deploy"
