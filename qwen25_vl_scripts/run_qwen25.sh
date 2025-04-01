# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_clean_image ./adversarial_images/clean.jpeg

# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_clean_image_add_noise_G30 ./adversarial_images_add_noise_G30/clean.jpeg
# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_clean_image_add_noise_G50 ./adversarial_images_add_noise_G50/clean.jpeg
# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_clean_image_add_noise_G75 ./adversarial_images_add_noise_G75/clean.jpeg

# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_adv_img_16 ./adversarial_images/prompt_constrained_16.bmp
# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_adv_img_32 ./adversarial_images/prompt_constrained_32.bmp
# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_adv_img_64 ./adversarial_images/prompt_constrained_64.bmp
# bash qwen25_vl_scripts/qwen25_vl_rtp.sh ./results/qwen25vl_adv_img_un ./adversarial_images/prompt_unconstrained.bmp

bash qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh ./results/qwen25vl_adv_img_16_diffpure_50 ./adversarial_images/prompt_constrained_16.bmp 50
bash qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh ./results/qwen25vl_adv_img_32_diffpure_50 adversarial_images/prompt_constrained_32.bmp 50
bash qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh ./results/qwen25vl_adv_img_64_diffpure_50 adversarial_images/prompt_constrained_64.bmp 50

bash qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh ./results/qwen25vl_adv_img_16_diffpure_100 adversarial_images/prompt_constrained_16.bmp 100
bash qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh ./results/qwen25vl_adv_img_32_diffpure_100 adversarial_images/prompt_constrained_32.bmp 100
bash qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh ./results/qwen25vl_adv_img_64_diffpure_100 adversarial_images/prompt_constrained_64.bmp 100