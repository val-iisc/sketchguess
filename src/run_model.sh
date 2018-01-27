
mkdir -p out
THEANO_FLAGS='device=gpu0,lib.cnmem=0.1,dnn.enabled=False' python -u src/test_model.py 2>&1 | tee out/lstm1lyr_512hid_predicitions.log
python -u src/generate_predictions.py > out/machine_generated_guesses.log
