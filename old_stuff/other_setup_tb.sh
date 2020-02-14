
# fix duplicate tensorboard version issue
#pip uninstall tb-nightly tensorboard tensorflow-estimator tensorflow-estimator-2.0-preview tensorflow-gpu tf-estimator-nightly tf-nightly-2.0-preview tf-nightly-gpu

#pip install tensorflow-gpu==2.0.0

# [tensorboard setup]
# [remote] run tensorboard:  --logdir logs --port 6006 --bind-all &
# [remote] jupyter notebook --ip=127.0.0.1 --port=8080 --allow-root &
# [local] ssh -N -f -L localhost:16006:localhost:6006 -p 28442 root@ssh5.vast.ai
