import org.apache.spark.sql.{ SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.functions._

object TopicModeling {

    def main(args: Array[String]): Unit = {
      if (args.length != 2) {
        println("Usage: InputFile OutputDir")
      }
      val spark = SparkSession.builder().appName("TopicModeling").getOrCreate()
      val book = spark.read.textFile(args(0)).toDF("text")
      val tokenizer = new RegexTokenizer().setPattern("[\\W_]+").setMinTokenLength(4).setInputCol("text").setOutputCol("textOut")
      val tokenized = tokenizer.transform(book)
      val remover = new StopWordsRemover().setInputCol("textOut").setOutputCol("filtered")
      val filtered_df = remover.transform(tokenized)
      val cv = new CountVectorizer()
        .setInputCol("filtered")
        .setOutputCol("features")
        .setVocabSize(10000)
        .setMinTF(2)
        .setMinDF(2)
        .setBinary(true)
      val cvFitted = cv.fit(filtered_df)
      val prepped = cvFitted.transform(filtered_df)
      val lda = new LDA().setK(5).setMaxIter(10)

      val model = lda.fit(prepped)
      val vocabList = cvFitted.vocabulary
      val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }

      val topics = model.describeTopics(maxTermsPerTopic = 5).withColumn("terms", termsIdx2Str(col("termIndices")))
      topics.select("topic", "terms", "termWeights").rdd.map(_.toString().replace("[", "").replace("]", "")).saveAsTextFile(args(1))
    }
}
