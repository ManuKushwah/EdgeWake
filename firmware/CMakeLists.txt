# EdgeWake firmware - ESP-IDF project (tested with ESP-IDF v5.5, target esp32s3)
cmake_minimum_required(VERSION 3.16)

# Embedded in the app descriptor; the OTA updater refuses images with the same version.
set(PROJECT_VER "1.0.0")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(edgewake)
