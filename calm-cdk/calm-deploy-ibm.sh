#!/usr/bin/env bash
# ibmcloud plugin install cloud-object-storage; ibmcloud login
set -euo pipefail

# Configuration variables (adjust or export these before running)
RESOURCE_GROUP_ID="${RESOURCE_GROUP_ID:-your_resource_group_id}"
VPC_ID="${VPC_ID:-your_vpc_id}"
SUBNET_ID="${SUBNET_ID:-your_subnet_id}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-your_security_group_id}"
SSH_KEY_ID="${SSH_KEY_ID:-your_ssh_key_id}"
SSH_PUBLIC_KEY_FILE="${SSH_PUBLIC_KEY_FILE:-$HOME/.ssh/id_rsa.pub}"
SSH_PRIVATE_KEY_FILE="${SSH_PRIVATE_KEY_FILE:-$HOME/.ssh/id_rsa}"
LOCAL_SRC_DIR="${LOCAL_SRC_DIR:-../../}"
COS_BUCKET_NAME="alpine-image-bucket"
ALPINE_VERSION="3.20.0"
IMAGE_FILENAME="nocloud_alpine-${ALPINE_VERSION}-aarch64-uefi-cloudinit-r0.qcow2"
IMAGE_URL="https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/cloud/${IMAGE_FILENAME}"

echo "Targeting IBM Cloud region and resource group..."
ibmcloud target -g "$RESOURCE_GROUP_ID"

echo "Downloading image..."
curl -L -o "$IMAGE_FILENAME" "$IMAGE_URL"

echo "Creating Cloud Object Storage bucket..."
ibmcloud cos bucket-create --bucket "$COS_BUCKET_NAME" || true

echo "Uploading image to Cloud Object Storage..."
ibmcloud cos object-put --bucket "$COS_BUCKET_NAME" --file "$IMAGE_FILENAME" --key "$IMAGE_FILENAME"

echo "Importing custom image into VPC..."
# Ensure jq is used for reliable JSON extraction if available, falling back safely
if command -v jq &> /dev/null; then
    CUSTOM_IMAGE_ID=$(ibmcloud is image-create "alpine-${ALPINE_VERSION}-arm64-uefi" \
        --file "cos://${COS_BUCKET_NAME}/${IMAGE_FILENAME}" \
        --os-name "alpine-3.20-aarch64" \
        --output json | jq -r '.id')
else
    IMAGE_IMPORT_RESPONSE=$(ibmcloud is image-create "alpine-${ALPINE_VERSION}-arm64-uefi" \
        --file "cos://${COS_BUCKET_NAME}/${IMAGE_FILENAME}" \
        --os-name "alpine-3.20-aarch64" \
        --output json)
    CUSTOM_IMAGE_ID=$(echo "$IMAGE_IMPORT_RESPONSE" | grep -o '"id": "[^"]*' | head -1 | cut -d'"' -f4)
fi

echo "Waiting for custom image to become available..."
ibmcloud is image-wait "$CUSTOM_IMAGE_ID"
rm -f "$IMAGE_FILENAME"

echo "Fetching zone for instance..."
if command -v jq &> /dev/null; then
    ZONE=$(ibmcloud is subnet "$SUBNET_ID" --output json | jq -r '.zone.name // empty')
fi
if [ -z "${ZONE:-}" ]; then
    ZONE=$(ibmcloud is subnet "$SUBNET_ID" --output json | grep -o '"name": "[^"]*' | grep -v 'subnet' | head -1 | cut -d'"' -f4)
fi
if [ -z "${ZONE:-}" ]; then
    ZONE="us-south-1" # Fallback default zone if parsing fails
fi

echo "Launching virtual server instance (ARM64 Ampere Altra)..."
if command -v jq &> /dev/null; then
    INSTANCE_RESPONSE=$(ibmcloud is instance-create \
        "alpine-app-instance-$(date +%s)" \
        "$VPC_ID" \
        "$ZONE" \
        "bz2-2x8" \
        "$SUBNET_ID" \
        --image "$CUSTOM_IMAGE_ID" \
        --keys "$SSH_KEY_ID" \
        --output json)
    INSTANCE_ID=$(echo "$INSTANCE_RESPONSE" | jq -r '.id')
else
    INSTANCE_RESPONSE=$(ibmcloud is instance-create \
        "alpine-app-instance-$(date +%s)" \
        "$VPC_ID" \
        "$ZONE" \
        "bz2-2x8" \
        "$SUBNET_ID" \
        --image "$CUSTOM_IMAGE_ID" \
        --keys "$SSH_KEY_ID" \
        --output json)
    INSTANCE_ID=$(echo "$INSTANCE_RESPONSE" | grep -o '"id": "[^"]*' | head -1 | cut -d'"' -f4)
fi

echo "Waiting for instance to run..."
ibmcloud is instance-wait "$INSTANCE_ID"

echo "Reserving and associating floating IP..."
if command -v jq &> /dev/null; then
    NIC_ID=$(ibmcloud is instance "$INSTANCE_ID" --output json | jq -r '.network_interfaces[0].id')
else
    NIC_ID=$(ibmcloud is instance "$INSTANCE_ID" --output json | grep -o '"id": "[^"]*' | head -2 | tail -1)
fi

FLOATING_IP_RESPONSE=$(ibmcloud is floating-ip-reserve "alpine-fip-$(date +%s)" \
    --nic "$NIC_ID" \
    --in "$INSTANCE_ID" \
    --output json)

if command -v jq &> /dev/null; then
    PUBLIC_IP=$(echo "$FLOATING_IP_RESPONSE" | jq -r '.address')
else
    PUBLIC_IP=$(echo "$FLOATING_IP_RESPONSE" | grep -o '"address": "[^"]*' | head -1 | cut -d'"' -f4)
fi

while [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "null" ]; do
    sleep 5
    if command -v jq &> /dev/null; then
        PUBLIC_IP=$(ibmcloud is instance "$INSTANCE_ID" --output json | jq -r '.floating_ips[0].address // empty')
    else
        PUBLIC_IP=$(ibmcloud is instance "$INSTANCE_ID" --output json | grep -o '"address": "[^"]*' | tail -1 || true)
    fi
done

echo "Waiting for SSH..."
REMOTE_USER="alpine"
until ssh -i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$REMOTE_USER@$PUBLIC_IP" "echo 'ready'" 2>/dev/null; do
    sleep 5
done

echo "Preparing remote dir..."
ssh -i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=no "$REMOTE_USER@$PUBLIC_IP" "mkdir -p ~/app"

echo "Copying source files..."
rsync -avz --exclude="deploy_alpine_ibmcloud.sh" --exclude=".git" -e "ssh -i $SSH_PRIVATE_KEY_FILE -o StrictHostKeyChecking=no" "$LOCAL_SRC_DIR" "$REMOTE_USER@$PUBLIC_IP:~/app/"

echo "Provisioning container runtime..."
ssh -i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=no "$REMOTE_USER@$PUBLIC_IP" << 'EOF'
    doas apk update
    doas apk add docker git make
    doas rc-update add docker boot
    doas service docker start
    doas addgroup alpine docker
    cd ~/app
    doas docker build -t remote-alpine-app .
    doas docker rm -f running-alpine-app || true
    doas docker run -d --name running-alpine-app -p 80:80 remote-alpine-app
EOF

echo "Done!"
