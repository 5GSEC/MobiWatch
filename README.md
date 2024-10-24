# MobiWatch

MobiWatch is an O-RAN compliant xApp that employs unsupervised unsupervised deep learning to detect layer-3 (RRC and NAS) cellular anomalies and attacks in 5G networks. MobiWatch operates on the security data telemetry called [MobiFlow](https://github.com/5GSEC/MobiFlow-Auditor), a security audit trail for holding mobile devices accountable during the link and session setup protocols as they interact with the base station.

Currently it is compatible with two nRT-RIC implmentations: [OSC RIC](https://lf-o-ran-sc.atlassian.net/wiki/spaces/ORAN) and [SD-RAN ONOS RIC](https://docs.sd-ran.org/master/sdran-in-a-box/README.html). You can deploy and test MobiWatch based on the [tutorial](https://github.com/5GSEC/5G-Spector/wiki/O%E2%80%90RAN-SC-RIC-Deployment-Guide) we have created to instantiate an O-RAN compliant 5G network with just open-sourced software such as [OpenAirInterface](https://gitlab.eurecom.fr/oai/openairinterface5g/). 

For more design details, please refer to our HotNets'24 research paper [6G-XSec: Explainable Edge Security for Emerging OpenRAN Architectures](). 

![alt text](./fig/sys.png)

## Prerequisite

### Local Docker registry

MobiWatch is built from source as a local Docker container. Refer to the official tutorial (https://docs.docker.com/engine/install/) to install and set up the Docker environment.

Create a local docker registry to host docker images: 

```
sudo docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

### OSC nRT RIC

Before deploying the xApp, make sure the OSC nRT-RIC is deployed by following this [tutorial](https://github.com/5GSEC/5G-Spector/wiki/O%E2%80%90RAN-SC-RIC-Deployment-Guide#deploy-the-osc-near-rt-ric).


### MobiFlow Auditor xApp

MobiWatch directly acquires security telemetry from the SDL generated from the [MobiFlow Auditor xApp](https://github.com/5GSEC/MobiFlow-Auditor) xApp. Following the instructions to prepare the environment and collect data from a 5G network.


## Build

Run the build script:

```
./build.sh
```


## Install / Uninstall the xApp

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


To uninstall MobiWatch from the Kubernetes cluster:

```
./undeploy.sh
```

## Running Example

MobiWatch's classification results with benign 5G network traffic:

```
[INFO 2024-10-23 21:42:23,990 dlagent.py:222]
    rnti        tmsi                     msg
0  60786  1450744508         RRCSetupRequest
1  60786  1450744508                RRCSetup
2  60786  1450744508        RRCSetupComplete
3  60786  1450744508     Registrationrequest
4  60786  1450744508   Authenticationrequest
5  60786  1450744508  Authenticationresponse
[INFO 2024-10-23 21:42:23,990 dlagent.py:223] Benign


[INFO 2024-10-23 21:42:23,993 dlagent.py:222]
    rnti        tmsi                     msg
1  60786  1450744508                RRCSetup
2  60786  1450744508        RRCSetupComplete
3  60786  1450744508     Registrationrequest
4  60786  1450744508   Authenticationrequest
5  60786  1450744508  Authenticationresponse
6  60786  1450744508     Securitymodecommand
[INFO 2024-10-23 21:42:23,993 dlagent.py:223] Benign
```

MobiWatch's classification results with an specific 5G network attack:

```
[ERROR 2024-10-24 16:07:40,227 dlagent.py:225]
    rnti        tmsi                    msg
0  53496  1450744508        RRCSetupRequest
1  53496  1450744508               RRCSetup
2  53496  1450744508       RRCSetupComplete
3  53496  1450744508    Registrationrequest
4  53496  1450744508  Authenticationrequest
5  53496  1450744508       Identityresponse
[ERROR 2024-10-24 16:07:40,227 dlagent.py:226] Abnormal
```

This attack represents a downlink overshadowing where the network's Authentication Request message is overwritten and the UE responds with an IdentityResponse message with its identity, constituding an identity extraction attack. MobiWatch classifies this as an abnormal event as it deviates from normal traffic the DL model was traineed on.


## Dataset

Datasets used for training the DL model are available in this [folder](./dataset). We provide both the original [pcap](./dataset/pcap/) format of the benign / attack traffic we have collected in a test 5G network based on OAI, as well as the [MobiFlow](https://github.com/5GSEC/MobiFlow-Auditor) security telemetry format in `.csv` (converted from the `.pcap` files) that are used to train our DL detection models.


## Model Training

MobiWatch has pre-trained DL models on benign layer-3 5G network traffic, including a vanilla [Autoencoder](./src/ai/autoencoder/model.py) model and a sequential [LSTM](./src/ai/deeplog/deeplog.py) model implemented by the [DeepLog](https://dl.acm.org/doi/pdf/10.1145/3133956.3134015) paper (CCS'17) and this [code repository](https://github.com/wuyifan18/DeepLog). The pre-trained models will be loaded into the xApp container.


## Publication

Please cite our research papers if you develop any products and prototypes based on our code and datasets:

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