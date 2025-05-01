docker exec -it oisin_echonetlvh bash


CUDA_VISIBLE_DEVICES=4 python /ultrasound-toolbox/usbmd/data/convert/echonetlvh.py --batch Batch1
CUDA_VISIBLE_DEVICES=5 python /ultrasound-toolbox/usbmd/data/convert/echonetlvh.py --batch Batch2
CUDA_VISIBLE_DEVICES=6 python /ultrasound-toolbox/usbmd/data/convert/echonetlvh.py --batch Batch3
CUDA_VISIBLE_DEVICES=7 python /ultrasound-toolbox/usbmd/data/convert/echonetlvh.py --batch Batch4

# for tomorrow:
CUDA_VISIBLE_DEVICES=4 python /ultrasound-toolbox/usbmd/data/convert/echonetlvh.py --batch Batch3
CUDA_VISIBLE_DEVICES=5 python /ultrasound-toolbox/usbmd/data/convert/echonetlvh.py --batch Batch4


# not a hdf5
/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonetlvh_v2025/train/0X124F13A3139A8495.hdf5