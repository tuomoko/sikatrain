PYTHONPATH="/home/tuomo/models/research:/home/tuomo/models/research/slim" python sika_api.py \
    --frozen_model_filename=/home/tuomo/sikatrain/model/graph/frozen_inference_graph.pb \
    --label_map_path=/home/tuomo/sikatrain/model/pig_label_map.pbtxt \
    --num_labels=6

