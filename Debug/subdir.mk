################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../MQDF_test.cpp \
../MQDF_training.cpp \
../Matrix.cpp \
../Transform.cpp \
../main.cpp 

OBJS += \
./MQDF_test.o \
./MQDF_training.o \
./Matrix.o \
./Transform.o \
./main.o 

CPP_DEPS += \
./MQDF_test.d \
./MQDF_training.d \
./Matrix.d \
./Transform.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


