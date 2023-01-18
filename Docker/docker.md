# Docker <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Images](#images)
- [Containers](#containers)
- [Volumes](#volumes)
- [Networks](#networks)
  - [localhost Connection to Container](#localhost-connection-to-container)
  - [Container Connection to Container](#container-connection-to-container)
  - [Building a Network](#building-a-network)
- [Dockerfile](#dockerfile)
- [Build Image](#build-image)
- [Run Docker Container](#run-docker-container)
- [Docker Compose](#docker-compose)

## Images

- Blueprints/templates for containers; they are read-only and contain the application as well as the necessary environment.
- Do not run themselves, but can be executed as containers.
- Images are created in layers, which efficiently rebuild and share images.

  |                                  command | definition                                         |
  | ---------------------------------------: | :------------------------------------------------- |
  |                          `docker images` | lists all docker images                            |
  |             `docker image inspect IMAGE` | displays configuration for the image               |
  |                       `docker rmi IMAGE` | removes the image if not being used by a container |
  |                     `docker image prune` | removes all images nto being used by a container   |
  | `docker tag original_image cloned_image` | clones original image                              |

## Containers

- Running, isolated instances of images, by adding a thin read-write layer on top of the image.
- Multiple containers can be created/started based on one and the same image.

|                        command | definition                                    |
| -----------------------------: | :-------------------------------------------- |
|                    `docker ps` | lists all running containers                  |
|                 `docker ps -a` | list all containers (including stopped ones)  |
|       `docker start CONTAINER` | starts container in detached mode by default  |
| `docker start -a -i CONTAINER` | starts container in attached/interactive mode |
|      `docker attach CONTAINER` | attaches to a detached container              |
|        `docker logs CONTAINER` | displays logged output from container         |
|        `docker stop CONTAINER` | stops the container if running                |
|          `docker rm CONTAINER` | removes container if not running              |
|       `docker container prune` | removes all non-running containers            |

- copies `local_folder` and all its contents to the `container_folder` in the container

```shell
docker cp local_folder/. CONTAINER:/container_folder
```

- copies `container_folder` in the container and its contents to the `local_folder`

```shell
docker cp CONTAINER:/container_folder local_folder
```

## Volumes

|                        command | definition                            |
| -----------------------------: | :------------------------------------ |
|             `docker volume ls` | lists all available volumes           |
| `docker volume inspect VOLUME` | displays configuration for the volume |
|  `docker volume create VOLUME` | manually creates a volume             |
|      `docker volume rm VOLUME` | removes volume                        |
|          `docker volume prune` | removes all anonymous volumes         |

- uses a named volume when running a docker container to store persisting data

```shell
# creates a named docker volume
docker run -v VOLUME:/app/container_data_folder IMAGE

# creates a binding mount to the local machine
docker run -v "path/on/local:/app" IMAGE

# creates a binding mount to the local machines current directory
# changes made locally will be reflected in the container
docker run -v "$(pwd):/app" IMAGE

# creates a binding mount to the local machine in read only mode
# docker is no longer in read/write mode and can't make changes
docker run -v "$(pwd):/app:ro" IMAGE
```

## Networks

### localhost Connection to Container

- when working with localhost in a docker container, change the `localhost` variable to docker's address domain understood by docker: `host.docker.internal` in your code.

  - this translates the IP address of your local host machine as seen by the docker container

- when using a database in a docker container, and working on an `backend` application on your local machine:
  - run in detached mode `-d` to run database container in the background.
  - publish the container's database default port when running the container to allow the backend application to access the database.

```shell
# postgres database container
docker run --name postgresdb --rm -d -p 5432:5432 postgres

# mongo database container
docker run --name mongodb --rm -d -p 27017:27017 mongo
```

- when using a backend application in a docker container, and working on a `frontend` application on your local machine:
  - run in detached mode `-d` to run backend container in the background.
  - publish the container backend port when running the container to allow the frontend application to access the backend.

```shell
# python backend container
docker run --name api-backend --rm -d -p 8000:8000 python-api
```

### Container Connection to Container

- when connecting a database container to another container, use `docker inspect CONTAINER` to find the IP address of the database container.dock

- when using a frontend application in a docker container, and working with a `backend` application in a container:
  - publish the port of the backend container to allow the frontend container from accessing the backend.
  - publish the port of the frontend container to allow the local machine to access the browser portion of the appliaction
    - _this may only be required with react applications due to js being ran in the browser instead of the server._

### Building a Network

|                              command | definition                                                          |
| -----------------------------------: | :------------------------------------------------------------------ |
|      `docker network create NETWORK` | creates a network for multi-containers.                             |
|                  `docker network ls` | lists all available networks.                                       |
| `docker run --network NETWORK IMAGE` | builds and launches a docker container on a created docker network. |

When using containers in a network, instead of `localhost` or `host.docker.internal`, use the name of the container and docker will automatically resolve.

```python
# typical url connection to a localhost database from an application
url_local = 'mongodb://localhost:27017/swfavorites'

# url connection to a localhost database from an application inside of a docker container
url_local_docker = 'mongodb://host.docker.internal:27017/swfavorites'

# url connection to a docker container database named 'mongodb' from an application in another container on the same docker network
url_mongo_container = 'mongodb://mongodb:27017/swfavorites'
```

## Dockerfile

- the `Dockerfile` should be setup as optimally as possible to avoid a change in a layer running a subsequent layer not needing to be re-run.

```docker

# tells docker which base image to use for build.
FROM python:3.10

# sets the working directory to /app so commands are executed there.
WORKDIR /app

# copies dependencies file to the working directory.
COPY requirements.txt /app

# adds dependencies needed for the image.
RUN pip install -r requirements.txt

# copies the remaining code in local directory to the /app directory in the image.
COPY . /app

# sets which port docker will expose to the local machine as environment variable
ENV PORT=8000

# documents which docker image port is exposed to the local machine
EXPOSE $PORT

# command ran when a container is built from the image.
CMD ["python", "app.py"]
```

## Build Image

```docker
# builds the image based on the Dockerfile in the current directory
docker build -t image-name .

-t: assigns a name:tag to an image
```

## Run Docker Container

```docker
# builds and runs container with local_port:docker_port configuration
docker run -p 5432dock:5432 IMAGE

--name: names the container instead of random naming
-p : identifies local-host-port:docker-port information
-d : runs docker container in detached mode
-it: runs docker in an interactive mode
--rm: automatically remove container when stopped

# stops the docker container
docker stop CONTAINER
```

- runs the container using an .env file for environment variables

```shell
docker run -p 9000:80 --env-file ./.env
```

## Docker Compose

|               command | definition                                           |
| --------------------: | :--------------------------------------------------- |
|   `docker-compose up` | runs containers/volumes/etc. in `docker-compose.yml` |
| `docker-compose down` | stops running docker compose                         |
|                       |                                                      |

- use the `build` key in `docker-compose.yml` to specify the location of a Dockerfile to build an image for a service.

- use the `image` key in `docker-compose.yml` to specify an already created image for the service.

- the `docker run` flags `-d` and `-rm` are not needed in `docker-compose.yml`.

  - the container is removed by default when using docker compose
  - run the container in detached by running `docker-compose up -d`.

- specifying a network when running docker compose is not required

  - docker compose will automatically create a new environment for all services specified inside `docker-compose.yml` and add them to that environment.
  - can still specify the networks the service should be accessible to by using the `networks` key in `docker-compose.yml`.

- for any named volumes being used by your services, you must add the named volume to the `volumes` top-level key in `docker-compose.yml`.
- can delete the volumes used by the docker compose by running `docker-compose down -v`.

- when specifying a bind mount in the `volumes` key of a service, you can use a relative path from the `docker-compose.yml` file.

  - bind mounts are not specified in the top-level `volumes` key.

- anonymous volumes are not specified in the top-level `volumes` key.
