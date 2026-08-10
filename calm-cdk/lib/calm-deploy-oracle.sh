#!/usr/bin/env bash
set -euo pipefail

# Configuration variables (adjust or export these before running)
COMPARTMENT_ID="${COMPARTMENT_ID:-ocid1.compartment.oc1..your_compartment_ocid}"
SSH_PUBLIC_KEY_FILE="${SSH_PUBLIC_KEY_FILE:-$HOME/.ssh/id_rsa.pub}"
SSH_PRIVATE_KEY_FILE="${SSH_PRIVATE_KEY_FILE:-$HOME/.ssh/id_rsa}"
LOCAL_SRC_DIR="${LOCAL_SRC_DIR:-../}"
BUCKET_NAME="alpine-image-bucket"
ALPINE_VERSION="3.20.0"
IMAGE_FILENAME="nocloud_alpine-${ALPINE_VERSION}-aarch64-uefi-cloudinit-r0.qcow2"
IMAGE_URL="https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/cloud/${IMAGE_FILENAME}"

echo "Fetching namespace..."
NAMESPACE=$(oci os ns get --query "data" --raw-output)

echo "Downloading image..."
curl -L -o "$IMAGE_FILENAME" "$IMAGE_URL"

echo "Creating bucket..."
oci os bucket create \
    --compartment-id "$COMPARTMENT_ID" \
    --name "$BUCKET_NAME" \
    --namespace-name "$NAMESPACE" || true

echo "Uploading image..."
oci os object put \
    --namespace-name "$NAMESPACE" \
    --bucket-name "$BUCKET_NAME" \
    --file "$IMAGE_FILENAME" \
    --name "$IMAGE_FILENAME"

echo "Importing image..."
IMAGE_IMPORT_RESPONSE=$(oci compute image import from-object \
    --compartment-id "$COMPARTMENT_ID" \
    --namespace "$NAMESPACE" \
    --bucket-name "$BUCKET_NAME" \
    --name "$IMAGE_FILENAME" \
    --display-name "alpine-${ALPINE_VERSION}-arm64-uefi" \
    --source-image-type QCOW2 \
    --wait-for-state AVAILABLE)

CUSTOM_IMAGE_ID=$(echo "$IMAGE_IMPORT_RESPONSE" | grep -o '"id": "[^"]*' | head -1 | cut -d'"' -f4)
rm -f "$IMAGE_FILENAME"

echo "Fetching subnet..."
SUBNET_ID=$(oci network subnet list --compartment-id "$COMPARTMENT_ID" --lifecycle-state AVAILABLE --query "data[0].id" --raw-output)

echo "Launching instance..."
AD=$(oci iam availability-domain list --compartment-id "$COMPARTMENT_ID" --query "data[0].name" --raw-output)
INSTANCE_RESPONSE=$(oci compute instance launch \
    --compartment-id "$COMPARTMENT_ID" \
    --availability-domain "$AD" \
    --shape "VM.Standard.A1.Flex" \
    --shape-config '{"ocpus": 4, "memoryInGBs": 24}' \
    --image-id "$CUSTOM_IMAGE_ID" \
    --subnet-id "$SUBNET_ID" \
    --assign-public-ip true \
    --ssh-authorized-keys-file "$SSH_PUBLIC_KEY_FILE" \
    --wait-for-state RUNNING)

INSTANCE_ID=$(echo "$INSTANCE_RESPONSE" | grep -o '"id": "[^"]*' | head -1 | cut -d'"' -f4)

echo "Getting IP..."
PUBLIC_IP=""
while [ -z "$PUBLIC_IP" ]; do
    sleep 5
    PUBLIC_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" --query 'data[0]."public-ip"' --raw-output 2>/dev/null || true)
done

echo "Waiting for SSH..."
REMOTE_USER="alpine"
until ssh -i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$REMOTE_USER@$PUBLIC_IP" "echo 'ready'" 2>/dev/null; do
    sleep 5
done

echo "Preparing remote dir..."
ssh -i "$SSH_PRIVATE_KEY_FILE" -o StrictHostKeyChecking=no "$REMOTE_USER@$PUBLIC_IP" "mkdir -p ~/app"

echo "Copying source files..."
rsync -avz --exclude="deploy_alpine_oci.sh" --exclude=".git" -e "ssh -i $SSH_PRIVATE_KEY_FILE -o StrictHostKeyChecking=no" "$LOCAL_SRC_DIR" "$REMOTE_USER@$PUBLIC_IP:~/app/"

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
