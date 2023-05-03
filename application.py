from flask import Flask, render_template, request
from recommender import recommend_nmf

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')
  
@app.route('/results')
def recom():
    user_query = request.args.to_dict()
    user_query = {key:int(value) for key,value in user_query.items()}
    top_two = recommend_nmf(user_query)
    #, user_query=user_query
    return render_template('results.html', top_two=top_two)
if __name__ == "__main__":
    app.run(debug=True, port=5001)