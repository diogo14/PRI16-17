import os
import networkx as nx
import json
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import xml.etree.ElementTree as ET


from util import removePunctuation
from util import getWordGrams
from util import printTopCandidates
from util import getOrderedCandidates
from util import writeToFile

root = ET.parse(os.path.join(os.path.dirname(__file__), "resources", "MiddleEast.xml"))
articles = root.findall('./channel/item')

data = ""

for article in articles:
    data += article.findall('./title')[0].text.lower() + "."
    data +=  article.findall('./description')[0].text.lower() + "."

data = data.encode('ascii', 'ignore')   #TODO better encoding


################ Keyphrase extraction part ####################

def calcPR(candidate, graph, candidate_scores):

    linked_candidates = graph.neighbors(candidate)   #set of candidates that co-occur with candidate
    number_linked_candidates = len(linked_candidates)    # |Links(Pj)|
    N = len(candidate_scores)            #number of candidates
    d = 0.5

    #print linked_candidates

    summatory = 0.0
    for neighbor_candidate in linked_candidates:
        summatory += candidate_scores[neighbor_candidate] / float(number_linked_candidates)

    return d/N + (1-d) * summatory

sentences = map(removePunctuation, PunktSentenceTokenizer().tokenize(data))   #with removed punctuation
n_grammed_sentences = [getWordGrams(nltk.word_tokenize(sentence), 1, 4) for sentence in sentences]

tokenized_document = nltk.word_tokenize(removePunctuation(data))
n_grammed_document = getWordGrams(tokenized_document, 1, 4)


g = nx.Graph()
g.add_nodes_from(n_grammed_document)

#adding edges to the undirected  unweighted graph (gram, another_gram) combinatins within the same sentence. for each sentence
for sentence in n_grammed_sentences:
     for gram in sentence:
         for another_gram in sentence:
             if another_gram == gram:
                 continue
             else:
                 g.add_edge(gram, another_gram) #adding duplicate edges has no effect

#initializing each candidate score to 1
candidate_PR_scores = {}
for candidate in n_grammed_document:
    candidate_PR_scores[candidate] = 1

#iterative converging PR score calculation
for i in range(0, 10):
    for candidate in n_grammed_document:
        score = calcPR(candidate, g, candidate_PR_scores)
        candidate_PR_scores[candidate] = score

################ Result Output #################################

printTopCandidates(candidate_PR_scores, 10)

ordered_candidates = getOrderedCandidates(candidate_PR_scores)

#adjusting candidate weights to show in word cloud
# candidate_score * maximum_score / MAXIMUM_WEIGHT
adjusted_candidate_weights = map(lambda x: {"text" : x[0], "size" : x[1] * 50 / ordered_candidates[0][1]}, ordered_candidates[:10])


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