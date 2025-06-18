temperature = 23.7      
humidity = 65.12346           
device_name = "TempSensor01"  
is_online = True  
      
print(f"Temperature : {temperature}");
print(f"Humidity: {humidity}");


print(temperature,humidity,device_name,is_online,sep=" | ");
print(temperature,humidity,device_name,is_online,sep=" , ");


print("Temperature: "+str(temperature));
print("Device: "+device_name);


print("Device is : {}C".format(is_online));
print("{:.1f}C {:.4f}".format(temperature,humidity))