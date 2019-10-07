# pipeline для препроцессинга изображений лиц

## Предварительные требования
	* docker
	* docker-compose
	* nvidia-docker

## Развёртывание
	1) git clone https://github.com/d-75/3dmm_pipeline.git
	2) ./build_docker_image.bsh (сборка docker-образа)
	3) ./download_data_docker.bsh (загрузка данных, необходимых для работы)
	4) ./build_pix2face_sources.bsh (сборка библиотек face3d и vxl)
