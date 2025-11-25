#!/bin/bash
VERSION="v01"
PROJECT="sssd_official"
mode=${1:-none}
HOSTNAME_C=$PROJECT"-"`hostname`
CONTAINER_NAME=$PROJECT"-"$VERSION
ORIGINAL_DATA_PATH="/export/work/data/deep_learning/nonaka"
PROCESSED_DATA_PATH="/export/work/users/nonaka/project/SynECG"
SYSTEM_DATA_PATH="/export/work/users/nonaka/project/SynECG/system_data"

if [ $mode = "buildx" ]; then
    DIRNAME=$(basename "`pwd`")
    CONTAINER_HOMEDIR="/home/$USER/$DIRNAME/"

    docker buildx build --load \
        --build-arg USERNAME="$USER" \
        --build-arg UID="$UID" \
        --build-arg GID="$(id -g "$USER")" \
        --build-arg ORIGINAL_DATA_PATH="$ORIGINAL_DATA_PATH" \
        --build-arg PROCESSED_DATA_PATH="$PROCESSED_DATA_PATH" \
        --build-arg SYSTEM_DATA_PATH="$SYSTEM_DATA_PATH" \
        --progress=plain \
        -t "$PROJECT:$VERSION" \
        .
    docker run \
        --gpus all \
        -v "$ORIGINAL_DATA_PATH:$ORIGINAL_DATA_PATH" \
        -v "$PROCESSED_DATA_PATH:$PROCESSED_DATA_PATH" \
        -v "$SYSTEM_DATA_PATH:$SYSTEM_DATA_PATH" \
        -v "$(pwd):$CONTAINER_HOMEDIR" \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash
elif [ $mode = "build" ]; then
    DIRNAME=$(basename "`pwd`")
    CONTAINER_HOMEDIR="/home/$USER/$DIRNAME/"

    docker buildx build \
        --build-arg USERNAME=$USER \
        --build-arg UID=$UID \
        --build-arg GID=$(id -g $USER) \
        --build-arg ORIGINAL_DATA_PATH=$ORIGINAL_DATA_PATH \
        --build-arg PROCESSED_DATA_PATH=$PROCESSED_DATA_PATH \
        --build-arg SYSTEM_DATA_PATH=$SYSTEM_DATA_PATH \
        --progress=plain \
        -t $PROJECT:$VERSION . < Dockerfile
    docker run \
        --gpus all \
        -v "$ORIGINAL_DATA_PATH:$ORIGINAL_DATA_PATH" \
        -v "$PROCESSED_DATA_PATH:$PROCESSED_DATA_PATH" \
        -v "$SYSTEM_DATA_PATH:$SYSTEM_DATA_PATH" \
        -v "$(pwd):$CONTAINER_HOMEDIR" \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash        
elif [ $mode = "start" ]; then
    docker run \
        --gpus all \
        -v "$ORIGINAL_DATA_PATH:$ORIGINAL_DATA_PATH" \
        -v "$PROCESSED_DATA_PATH:$PROCESSED_DATA_PATH" \
        -v "$SYSTEM_DATA_PATH:$SYSTEM_DATA_PATH" \
        -v "$(pwd):$CONTAINER_HOMEDIR" \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash
elif [ $mode = "restart" ]; then
    docker start $CONTAINER_NAME 
fi

docker exec -it $CONTAINER_NAME /bin/bash
