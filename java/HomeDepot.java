import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.lang.Iterable;
import java.util.Comparator;
import java.io.Serializable;

import scala.Tuple2;

import org.apache.commons.lang.StringUtils;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;


public class HomeDepot {
  public static void main(String[] args) throws Exception {
    String inputFile = args[0];
    String outputFile = args[1];

    // Create a Java Spark Context.
    SparkConf conf = new SparkConf().setAppName("HomeDepot");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load our input data.
    JavaRDD<String> input = sc.textFile(inputFile).repartition(1);

    // Split to product -> attribute pairs
    JavaPairRDD<String, String> commons = input.mapToPair(new PairFunction<String, String, String>() {
        public Tuple2<String, String> call(String x) {
        	String[] split = x.split("\"");
        	if(split.length < 4) return new Tuple2("", "");
        	String productId = "###" + split[0].replace(",", "");
        	String attribute = split[1] + " " + split[3];
        	return new Tuple2(productId, attribute);
        }
    });

    // Filter out empty keys
    commons = commons.filter(new Function<Tuple2<String, String>, Boolean>() {
    	public Boolean call(Tuple2<String, String> pair) {
    		return !("".equals(pair._1.trim()) || "".equals(pair._2.trim()));
    	}
    });

    // merge to a single product -> attribute pair for each product
    JavaPairRDD<String, String> all = commons.reduceByKey(new Function2<String, String, String>() {
    	public String call(String a1, String a2) {
    		return a1 + " " + a2;
    	}
    });

    // add double quote around values
    all = all.mapValues(new Function<String, String>() {
    	public String call(String val) {
    		return "\"" + val + "\"" + "###";
    	}
    });
   
  	// save to file
  	all.saveAsTextFile(outputFile);
  }
}
