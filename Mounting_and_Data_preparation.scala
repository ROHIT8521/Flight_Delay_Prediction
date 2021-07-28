// Databricks notebook source
val containerName = "margie-container"
val storageAccountName = "margiestorageacc"
val sas = "?sv=2020-08-04&ss=b&srt=sco&sp=rwdlactfx&se=2021-07-26T01:18:00Z&st=2021-07-25T17:18:00Z&spr=https&sig=UrngfcBDQywF%2Bu%2BVGxFMkZ9WNhyNnmNFypbYmwLKyk8%3D"
val config = "fs.azure.sas." + containerName+ "." + storageAccountName + ".blob.core.windows.net"

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://margie-container@margiestorageacc.blob.core.windows.net/FlightDelays",
  mountPoint = "/mnt/FlightDelays",
  extraConfigs = Map(config -> sas))

// COMMAND ----------

val flight_delay_df = spark.read
.option("header","true")
.option("inferSchema", "true")
.csv("/mnt/FlightDelays")
display(flight_delay_df)

// COMMAND ----------

flight_delay_df.createOrReplaceTempView("flight_delay_data")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from flight_delay_data

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) from flight_delay_data

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) from flight_delay_data where DepDel15 is null

// COMMAND ----------

// MAGIC %python
// MAGIC dfFlightDelays = spark.sql("select * from flight_delay_data")

// COMMAND ----------

// MAGIC %python
// MAGIC print(dfFlightDelays.dtypes)

// COMMAND ----------

// MAGIC %r
// MAGIC library(SparkR)
// MAGIC 
// MAGIC # Select only the columns we need, casting CRSDepTime as long and DepDel15 as int, into a new DataFrame
// MAGIC dfflights <- sql("SELECT OriginAirportCode, OriginLatitude, OriginLongitude, Month, DayofMonth, cast(CRSDepTime as long) CRSDepTime, DayOfWeek, Carrier, DestAirportCode, DestLatitude, DestLongitude, cast(DepDel15 as int) DepDel15 from flight_delay_data")
// MAGIC 
// MAGIC # Delete rows containing missing values
// MAGIC dfflights <- na.omit(dfflights)
// MAGIC 
// MAGIC # Round departure times down to the nearest hour, and export the result as a new column named "CRSDepHour"
// MAGIC dfflights$CRSDepHour <- floor(dfflights$CRSDepTime / 100)
// MAGIC 
// MAGIC # Trim the columns to only those we will use for the predictive model
// MAGIC dfflightsClean = dfflights[, c("OriginAirportCode","OriginLatitude", "OriginLongitude", "Month", "DayofMonth", "CRSDepHour", "DayOfWeek", "Carrier", "DestAirportCode", "DestLatitude", "DestLongitude", "DepDel15")]
// MAGIC 
// MAGIC createOrReplaceTempView(dfflightsClean, "flight_delays_view")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from flight_delays_view

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) from flight_delays_view

// COMMAND ----------

// MAGIC %python
// MAGIC dfFlightDelays_Clean = spark.sql("select * from flight_delays_view")

// COMMAND ----------

// MAGIC %python
// MAGIC dfFlightDelays_Clean.write.mode("overwrite").saveAsTable("flight_delays_clean")

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://margie-container@margiestorageacc.blob.core.windows.net/FlightWeather",
  mountPoint = "/mnt/FlightWeather",
  extraConfigs = Map(config -> sas))

// COMMAND ----------

val flight_weather_df = spark.read
.option("header","true")
.option("inferSchema", "true")
.csv("/mnt/FlightWeather")
display(flight_weather_df)

// COMMAND ----------

flight_weather_df.createOrReplaceTempView("flight_weather_data")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from flight_weather_data

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) from flight_weather_data

// COMMAND ----------

// MAGIC %sql
// MAGIC select distinct WindSpeed from flight_weather_data

// COMMAND ----------

// MAGIC %sql
// MAGIC select distinct SeaLevelPressure from flight_weather_data

// COMMAND ----------

// MAGIC %sql
// MAGIC select distinct HourlyPrecip from flight_weather_data

// COMMAND ----------

// MAGIC %python
// MAGIC dfWeather = spark.sql("select AirportCode, cast(Month as int) Month, cast(Day as int) Day, cast(Time as int) Time, WindSpeed, SeaLevelPressure, HourlyPrecip from flight_weather_data")

// COMMAND ----------

// MAGIC %python
// MAGIC dfWeather.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import functions as F

// COMMAND ----------

// MAGIC %python
// MAGIC # Round Time down to the next hour, since that is the hour for which we want to use flight data. Then, add the rounded Time to a new column named "Hour", and append that column to the dfWeather DataFrame.
// MAGIC df = dfWeather.withColumn('Hour', F.floor(dfWeather['Time']/100))
// MAGIC 
// MAGIC # Replace any missing HourlyPrecip and WindSpeed values with 0.0
// MAGIC df = df.fillna('0.0', subset=['HourlyPrecip', 'WindSpeed'])
// MAGIC 
// MAGIC # Replace any WindSpeed values of "M" with 0.005
// MAGIC df = df.replace('M', '0.005', 'WindSpeed')
// MAGIC 
// MAGIC # Replace any SeaLevelPressure values of "M" with 29.92 (the average pressure)
// MAGIC df = df.replace('M', '29.92', 'SeaLevelPressure')
// MAGIC 
// MAGIC # Replace any HourlyPrecip values of "T" (trace) with 0.005
// MAGIC df = df.replace('T', '0.005', 'HourlyPrecip')
// MAGIC 
// MAGIC # Be sure to convert WindSpeed, SeaLevelPressure, and HourlyPrecip columns to float
// MAGIC # Define a new DataFrame that includes just the columns being used by the model, including the new Hour feature
// MAGIC dfWeather_Clean = df.select('AirportCode', 'Month', 'Day', 'Hour', df['WindSpeed'].cast('float'), df['SeaLevelPressure'].cast('float'), df['HourlyPrecip'].cast('float'))

// COMMAND ----------

// MAGIC %python
// MAGIC display(dfWeather_Clean)

// COMMAND ----------

// MAGIC %python
// MAGIC print(dfWeather_Clean.dtypes)

// COMMAND ----------

// MAGIC %python
// MAGIC dfWeather_Clean.write.mode("overwrite").saveAsTable("flight_weather_clean")

// COMMAND ----------

// MAGIC %python
// MAGIC dfWeather_Clean.select("*").count()

// COMMAND ----------

// MAGIC %python
// MAGIC dfFlightDelaysWithWeather = spark.sql("SELECT d.OriginAirportCode, \
// MAGIC                  d.Month, d.DayofMonth, d.CRSDepHour, d.DayOfWeek, \
// MAGIC                  d.Carrier, d.DestAirportCode, d.DepDel15, w.WindSpeed, \
// MAGIC                  w.SeaLevelPressure, w.HourlyPrecip \
// MAGIC                  FROM flight_delays_clean d \
// MAGIC                  INNER JOIN flight_weather_clean w ON \
// MAGIC                  d.OriginAirportCode = w.AirportCode AND \
// MAGIC                  d.Month = w.Month AND \
// MAGIC                  d.DayofMonth = w.Day AND \
// MAGIC                  d.CRSDepHour = w.Hour")

// COMMAND ----------

// MAGIC %python
// MAGIC display(dfFlightDelaysWithWeather)

// COMMAND ----------

// MAGIC %python
// MAGIC dfFlightDelaysWithWeather.write.mode("overwrite").saveAsTable("flight_delays_with_weather")

// COMMAND ----------

// MAGIC %md ## storing cleaned dataset to blob storage

// COMMAND ----------

val clean_data = spark.sql("""
SELECT * from flight_delays_with_weather
""")

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://margie-container@margiestorageacc.blob.core.windows.net/",
  mountPoint = "/mnt/result",
  extraConfigs = Map(config -> sas)
)

// COMMAND ----------

clean_data.write
 .option("header", "true")
 .format("com.databricks.spark.csv")
 .save("/mnt/result/Flight_Delay_Weather_Clean.csv")
