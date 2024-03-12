---
title: AImarkerspace Week 1
emoji: 🏢
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
license: openrail
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## How to build
docker build -t middleterm .

## How to run in local
docker run --name middleterm -p 7860:7860 middleterm
