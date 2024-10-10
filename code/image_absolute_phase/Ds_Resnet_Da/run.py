import os
     

   
# dls_danet_resnet34


for w1 in [1]:#The weight of phl3 
    for w2 in [1]:#The weight of wrapped phase4
        for w3 in [100]:#The weight of unwrapped phase

            os.system (f'python .../main_train.py\
                        --w1 {w1} --w2 {w2} --w3 {w3} \
                        --batch_size 8 --epochs 200 --learning_rate 0.001 \
                        --wandb_project_name "statue_1605"\
                        --root ".../statue_1605/" \
                        --model "Result/..."')




