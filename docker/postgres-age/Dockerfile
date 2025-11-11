FROM ankane/pgvector:latest

ARG AGE_VERSION=PG15

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.utf8
ENV LC_ALL=en_US.utf8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        locales \
        flex \
        git \
        bison \
        postgresql-server-dev-15 \
    && update-ca-certificates \
    && localedef -i en_US -f UTF-8 en_US.UTF-8 \
    && git clone --depth 1 --branch "${AGE_VERSION}" https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make PG_CONFIG=/usr/lib/postgresql/15/bin/pg_config install \
    && cd / \
    && rm -rf /tmp/age \
    && apt-get purge -y --auto-remove \
        build-essential \
        git \
        bison \
        flex \
    && rm -rf /var/lib/apt/lists/*

COPY docker/postgres-age/initdb.d/ /docker-entrypoint-initdb.d/

RUN chmod +x /docker-entrypoint-initdb.d/*.sh

