################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../MQDF_training.cpp \
../Matrix.cpp \
../Transform.cpp \
../main.cpp 

OBJS += \
./MQDF_training.o \
./Matrix.o \
./Transform.o \
./main.o 

CPP_DEPS += \
./MQDF_training.d \
./Matrix.d \
./Transform.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


