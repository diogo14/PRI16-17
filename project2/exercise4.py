import os
import json
import nltk
import urllib2
from nltk.tokenize.punkt import PunktSentenceTokenizer
import xml.etree.ElementTree as ET

from util import getWordGrams
from util import getOrderedCandidates
from util import createNotWeightedGraph
from util import getCandidatesfromDocumentSentences
from util import pagerank

def writeToFile(path, content):
    f = open(path, 'w')
    f.write(content)
    f.close()

url = 'http://www.nytimes.com/services/xml/rss/nyt/Europe.xml'
root = ET.parse(urllib2.urlopen(url))
articles = root.findall('./channel/item')

data = []
for article in articles:
    data.append(unicode(article.findall('./title')[0].text.lower()))
    data.append(unicode(article.findall('./description')[0].text.lower()))

################ Keyphrase extraction part ####################

sentences = []
[sentences.append(PunktSentenceTokenizer().tokenize(d)) for d in data]
n_grammed_sentences = [getWordGrams(nltk.word_tokenize(sentence[0]), 1, 4) for sentence in sentences]

document_candidates = getCandidatesfromDocumentSentences(n_grammed_sentences)

candidate_scores = pagerank(createNotWeightedGraph(document_candidates, n_grammed_sentences), 0, 0)

################ Result Output #################################

ordered_candidates = getOrderedCandidates(candidate_scores)

#adjusting candidate weights to show in word cloud
# candidate_score * maximum_score / MAXIMUM_WEIGHT
adjusted_candidate_weights = map(lambda x: {"text" : x[0], "size" : x[1] * 50 / ordered_candidates[0][1]}, ordered_candidates)


output_dir = os.path.join(os.path.dirname(__file__), "exercise4output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#writing to JSON file that is used as input to Word Cloud
writeToFile(os.path.join(output_dir, "data.json"), json.dumps(adjusted_candidate_weights))

#generate HTML if does not exist
if not os.path.isfile(os.path.join(os.path.dirname(output_dir), "index.html")):
    html = """<!DOCTYPE html>
    <html>
    <script src="http://d3js.org/d3.v3.min.js"></script>
    <script src="https://rawgit.com/jasondavies/d3-cloud/master/build/d3.layout.cloud.js"></script>
    <head>
        <title>Word Cloud Example</title>
    </head>

    <body>
    <script>

        var fill = d3.scale.category20();

        var SIZE_W = 800;
        var SIZE_H = 800;

         readJsonObject("data.json", function(text){
            var words_data = JSON.parse(text);
            console.log(words_data);

            d3.layout.cloud()
            .size([SIZE_W, SIZE_H])
            .words(words_data)
            .padding(5)
            .rotate(function() { return ~~(Math.random() * 2) * 90; })
            .font("Impact")
            .fontSize(function(d) { return d.size; })
            .on("end", draw)
            .start();
        });

        function draw(words) {
           d3.select("body").append("svg")
           .attr("width", SIZE_W)
           .attr("height", SIZE_H)
          .append("g")
          .attr("transform", "translate(" + SIZE_W / 2 + "," + SIZE_H / 2 + ")")
          .selectAll("text")
          .data(words)
          .enter().append("text")
          .style("font-size", function(d) { return d.size + "px"; })
          .style("font-family", "Impact")
          .style("fill", function(d, i) { return fill(i); })
          .attr("text-anchor", "middle")
          .attr("transform", function(d) {
            return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
          })
          .text(function(d) { return d.text; });
        }

        function readJsonObject(file, callback) {
            var rawFile = new XMLHttpRequest();
            rawFile.overrideMimeType("application/json");
            rawFile.open("GET", file, true);
            rawFile.onreadystatechange = function() {
                if (rawFile.readyState === 4 && rawFile.status == "200") {
                    callback(rawFile.responseText);
                }
            }
            rawFile.send(null);
        }
    </script>
    </body>
    </html>
    """

    writeToFile(os.path.join(output_dir, "index.html"), html)