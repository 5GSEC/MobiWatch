# MobiWatch

MobiWatch is an O-RAN compliant xApp that employs unsupervised unsupervised deep learning to detect layer-3 (RRC and NAS) cellular anomalies and attacks in 5G networks.

For more design details, please refer to our HotNets'24 research paper [6G-XSec: Explainable Edge Security for Emerging OpenRAN Architectures](). 

Currently it is compatible with two nRT-RIC implmentation: OSC RIC and SD-RAN ONOS RIC.

![alt text](./fig/sys.png)

## Prerequisite

Deploy OSC's nearRT RIC and the MobiFlow Auditor xApp before deploying MobiWatch. Detailed tutorial can be found at this [link](https://github.com/5GSEC/5G-Spector/wiki/O%E2%80%90RAN-SC-RIC-Deployment-Guide).

## Build

Build the MobiWatch xApp locally from source as a docker image. First setup the local docker registry:

```
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

Then run the build script:

```
./build.sh
```

## Dataset

## Model Training


## Install the MobieXpert xApp

First, onboard the xApp. You need to set up the proper environment with the `dms_cli` tool. Follow the instructions [here](https://github.com/5GSEC/5G-Spector/wiki/O%E2%80%90RAN-SC-RIC-Deployment-Guide) to install the tool. 

Then execute the following to onboard the xApp:

```
cd init
sudo -E dms_cli onboard --config_file_path=config-file.json --shcema_file_path=schema.json
```

Then, simply run the script to deploy the xApp under the `ricxapp` K8S namespace in the nRT-RIC.

```
cd ..
./deploy.sh
```

Successful deployment (this may take a while):

```
$ kubectl get pods -n ricxapp
ricxapp        ricxapp-mobiwatch-xapp-6b8879868d-fmnbd                      1/1     Running     0             5m32s
```

## Undeployment

Deploy MobiWatch from the Kubernetes cluster:

```
./undeploy.sh
```

## Citation

```
@inproceedings{6G-XSEC:Hotnets24,
  title     = {6G-XSec: Explainable Edge Security for Emerging OpenRAN Architectures },
  author    = {Wen, Haohuang and Sharma, Prakhar and Yegneswaran, Vinod and Porras, Phillip and Gehani, Ashish and Lin, Zhiqiang},
  booktitle = {Proceedings of the Twenty-Third ACM Workshop on Hot Topics in Networks (HotNets 2024)},
  address   = {Irvine, CA},
  month     = {November},
  year      = 2024
}
```