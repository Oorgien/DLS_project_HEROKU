# DLS project
 DLS project deployed on HEROKU

Для взаимодействия с ботом пропишите /start или /help.

Доступ к боту @Oorgien_bot

Связаться со мной можно в телеграме @Oorgien

Gan'ы реализованы при помощи готового фреймворка 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix,
но чтобы оно работало исправно, мне пришлось его немного поменять изнутри.

Код этих изменений есть в репозитории https://github.com/Oorgien/DLS_project. 

Чтобы все уместилось на HEROKU, пришлось GAN'ы сохранять в формате jit и немного менять архитектуру сети, чтобы все работало https://github.com/Oorgien/DLS_project/blob/423b8a13ffb41fc203ad392914182c7cb995d799/pytorch_CycleGAN_and_pix2pix/models/base_model.py#L256.
