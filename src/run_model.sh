THEANO_FLAGS='device=gpu0,lib.cnmem=0.1,dnn.enabled=False' python -u test_model.py 2>&1 | tee ../out/lstm1lyr_512hid_fseq.log
