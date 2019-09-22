# pipeline для препроцессинга изображений лиц

## Предварительные требования
	* docker
	* docker-compose

## Развёртка
	* git clone --recursive https://github.com/d-76/3dmm_pipeline.git (рекурсивно клонируем)
	* ./build_docker_image.bsh (сборка docker-образа)
	* ./download_data_docker.bsh (загрузка данных, необходимых для работы)
	* ./build_pix2face_sources.bsh (сборка библиотек face3d и vxl)