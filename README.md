# mmrotate-1x
## 环境配置
```python
# conda env
conda create -n rotate_itpn python=3.8 -y
conda activate rotate_itpn
# torch 1.11.0
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# mmrotate-1x
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate-1x
pip install -v -e .
# mmcv2.0.0rc4, 编译安装可以避免一些BUG
wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.0.0rc4.zip
unzip mmcv-2.0.0rc4.zip 
cd mmcv-2.0.0rc4/
pip install -v -e .
cd ..
# mmdet=3.0.0rc6
pip install mmdet==3.0.0rc6
# move rotated_fast_itpn_files to mmrotate-1x
git clone https://github.com/YangWei231/Rotated_fast_itpn.git
cp -r Rotated_fast_itpn/* ./
pip install -r requirements.txt
```
## 训练
dotav1-ss
```bash
# fast_itpn_base
bash tools/dist_train.sh configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_fast_itpn_base_1x_dota_0224.py 4
# fast_itpn_small
bash tools/dist_train.py configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_fast_itpn_small_1x_dota_0224.py 4
# fast_tipn_tiny
bash tools/dist_train.sh configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_fast_itpn_tiny_1x_dota_0224.py 4
```
dotav1-ss
```bash
# fast_itpn_base
bash tools/dist_train.sh configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_fast_itpn_base_1x_dota_0224-ms.py 4
# fast_itpn_small
bash tools/dist_train.py configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_fast_itpn_small_1x_dota_0224-ms.py 4
# fast_tipn_tiny
bash tools/dist_train.sh configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_fast_itpn_tiny_1x_dota_0224.py-ms 4
```



