for train_weight=0.0:0.01:0.10
    open("results/erw2_test_$(train_weight).sh", "w") do file
        write(file, """#!/bin/sh
              #SBATCH --gres=gpu:0
              #SBATCH --partition=Luna,Gobi
              #SBATCH --output=results/erw2_test_$(train_weight).out
              #SBATCH --job-name="erw2_test_$(train_weight)"

              python ../model/eval.py saved_models/erw2_train_$(train_weight) --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode erw2 --dataset test --model_desc erw2_test_$(train_weight)

              """)
    end
end

for train_weight=0.0:0.01:0.10
    open("results/erw1_test_$(train_weight).sh", "w") do file
        write(file, """#!/bin/sh
              #SBATCH --gres=gpu:0
              #SBATCH --partition=Luna,Gobi
              #SBATCH --output=results/erw1_test_$(train_weight).out
              #SBATCH --job-name="erw1_test_$(train_weight)"

              python ../model/eval.py saved_models/erw1_train_$(train_weight) --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode erw1 --dataset test --model_desc erw1_test_$(train_weight)

              """)
    end
end

open("results/finetune_attn_test.sh", "w") do file
  write(file, """#!/bin/sh
    #SBATCH --gres=gpu:0
    #SBATCH --partition=Luna,Gobi
    #SBATCH --output=results/finetune_attn_test
    #SBATCH --job-name="finetune_attn"

    python ../model/eval.py saved_models/finetune_attn_train --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode standard --dataset test --model_desc finetune_attn_test

    """)
end

open("results/joint_training_test.sh", "w") do file
  write(file, """#!/bin/sh
    #SBATCH --gres=gpu:0
    #SBATCH --partition=Luna,Gobi
    #SBATCH --output=results/joint_training_test
    #SBATCH --job-name="joint_training_test"

    python ../model/eval.py saved_models/joint_training_train --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode standard --dataset test --model_desc joint_training_test

    """)
end

