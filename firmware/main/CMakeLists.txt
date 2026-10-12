set(srcs
    "main.c"
    "audio_capture.c"
    "kws_task.c"
    "inference.cc"
    "actions.c"
    "probes.c"
    "wifi.c"
    "telemetry.c"
    "ota.c"
    "health.c"
    "model/model_data.cc"
    "model/model_settings.c")

# AWS IoT credentials are provisioned per device and never committed. Without
# them the firmware still builds and runs offline (telemetry disabled).
set(cert_dir "${CMAKE_CURRENT_SOURCE_DIR}/certs")
set(certs "${cert_dir}/AmazonRootCA1.pem" "${cert_dir}/device.pem.crt" "${cert_dir}/private.pem.key")
set(have_certs 1)
foreach(c ${certs})
    if(NOT EXISTS "${c}")
        set(have_certs 0)
    endif()
endforeach()
if(have_certs)
    set(embed_files ${certs})
else()
    message(WARNING "EdgeWake: AWS IoT certificates not found in main/certs - MQTT telemetry disabled")
    set(embed_files "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "." "model"
    PRIV_REQUIRES edgewake_core esp_driver_i2s esp_driver_gpio esp_wifi esp_netif esp_event
                  nvs_flash mqtt esp_https_ota esp_http_client app_update esp_timer json
                  esp_app_format mbedtls
    EMBED_TXTFILES ${embed_files})

target_compile_definitions(${COMPONENT_LIB} PRIVATE EW_HAVE_CERTS=${have_certs})
