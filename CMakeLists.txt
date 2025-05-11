cmake_minimum_required(VERSION 3.24)
project(PinkTrombone LANGUAGES CXX)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Указываем исходники
set(SOURCES
    pinktrombone.cpp
    simplex_noise.cpp
)

# Создаём исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES})

# Подключаем OpenAL
find_package(OpenAL REQUIRED)

if (OpenAL_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENAL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENAL_LIBRARY})
else()
    message(FATAL_ERROR "OpenAL not found")
endif()
