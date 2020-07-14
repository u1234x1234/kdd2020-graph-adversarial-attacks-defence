# kdd2020-graph-adversarial-attacks-defence

https://github.com/biendata-com/kdd2020_phase2_sample

docker build -t defe:001 ./

docker run --gpus=0 -it -v /home/u1234x1234/kdd2020-graph-adversarial-attacks-defence/data/kdd_cup_phase_two/:/data defe:001 "/data/adj_matrix_formal_stage.pkl /data/feature_formal_stage.npy /data/output.csv"
