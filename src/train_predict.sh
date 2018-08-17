# add stage1 test data to input/extra_data/stage1_train

python3.6 model_vector_unet.py train --model unet4_64_wshed_s1 --seed 41
python3.6 model_vector_unet.py train --model unet4_64_wshed_s2 --seed 42
python3.6 model_vector_unet.py train --model unet4_64_wshed_s3 --seed 43
python3.6 model_vector_unet.py train --model unet4_64_wshed_s4 --seed 44


python3.6 model_vector_unet.py prepare_submission --submission_name u7_128 --model unet7_64_wshed --use-tta --filter-masks --weights ../output/checkpoints/unet7_64_wshed_-1/checkpoint-128*
python3.6 model_vector_unet.py prepare_submission --submission_name u7_144 --model unet7_64_wshed --use-tta --filter-masks --weights ../output/checkpoints/unet7_64_wshed_-1/checkpoint-144*
python3.6 model_vector_unet.py prepare_submission --submission_name u7_160 --model unet7_64_wshed --use-tta --filter-masks --weights ../output/checkpoints/unet7_64_wshed_-1/checkpoint-160*


python3.6 model_vector_unet.py prepare_submission --submission_name s1_136 --model unet4_64_wshed_s1 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s1_-1/checkpoint-136*
python3.6 model_vector_unet.py prepare_submission --submission_name s1_160 --model unet4_64_wshed_s1 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s1_-1/checkpoint-160*

python3.6 model_vector_unet.py prepare_submission --submission_name s2_136 --model unet4_64_wshed_s2 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s2_-1/checkpoint-136*
python3.6 model_vector_unet.py prepare_submission --submission_name s2_160 --model unet4_64_wshed_s2 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s2_-1/checkpoint-160*

python3.6 model_vector_unet.py prepare_submission --submission_name s3_136 --model unet4_64_wshed_s3 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s3_-1/checkpoint-136*
python3.6 model_vector_unet.py prepare_submission --submission_name s3_160 --model unet4_64_wshed_s3 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s3_-1/checkpoint-160*

python3.6 model_vector_unet.py prepare_submission --submission_name s4_136 --model unet4_64_wshed_s4 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s4_-1/checkpoint-136*
python3.6 model_vector_unet.py prepare_submission --submission_name s4_160 --model unet4_64_wshed_s4 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_s4_-1/checkpoint-160*

# prepare the final ensemble
python3.6 model_vector_unet.py combine_submissions --submission_name sub_1 --use-tta --filter-masks --submissions_to_combine s1_136,s1_160,s2_136,s2_160,s3_136,s3_160,s4_136,s4_160,u7_128,u7_144,u7_160



python3.6 model_vector_unet.py train --model unet4_64_wshed_h1 --seed 51
python3.6 model_vector_unet.py train --model unet4_64_wshed_h2 --seed 52

python3.6 model_vector_unet.py prepare_submission --submission_name h1_136 --model unet4_64_wshed_h1 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_h1_-1/checkpoint-136*
python3.6 model_vector_unet.py prepare_submission --submission_name h1_160 --model unet4_64_wshed_h1 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_h1_-1/checkpoint-160*

python3.6 model_vector_unet.py prepare_submission --submission_name h2_136 --model unet4_64_wshed_h2 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_h2_-1/checkpoint-136*
python3.6 model_vector_unet.py prepare_submission --submission_name h2_160 --model unet4_64_wshed_h2 --use-tta --filter-masks --weights ../output/checkpoints/unet4_64_wshed_h2_-1/checkpoint-160*

# prepare the final ensemble2
# uncomment sub2 related section in config.py
python3.6 model_vector_unet.py combine_submissions --submission_name sub_2 --use-tta --filter-masks --submissions_to_combine h1_136,h1_144,h1_160,h2_136,h2_152,h2_160,s1_136,s1_160,s2_136,s2_160,s3_136,s3_160,s4_136,s4_160,u7_128,u7_144,u7_160
