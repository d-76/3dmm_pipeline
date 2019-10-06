# pipeline для препроцессинга изображений лиц

## Предварительные требования
	* docker
	* docker-compose
	* nvidia-docker

## Развёртка
	1) git clone --recursive https://github.com/d-76/3dmm_pipeline.git (рекурсивно клонируем)
	2) ./build_docker_image.bsh (сборка docker-образа)
	3) ./download_data_docker.bsh (загрузка данных, необходимых для работы)
	4) ./build_pix2face_sources.bsh (сборка библиотек face3d и vxl)
