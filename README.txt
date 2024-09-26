# Leveraging machine learning to streamline the development of liposomal drug delivery systems

Drug delivery systems efficiently and safely administer therapeutic agents to specific body sites. Liposomes, spherical vesicles made of phospholipid bilayers, have become a powerful tool in this field, especially with the rise of microfluidic manufacturing during the COVID-19 pandemic. Despite its efficiency, microfluidic liposomal production poses challenges, often requiring laborious, optimization on a case-by-case basis. This is due to a lack of comprehensive understanding and robust methodologies, compounded by limited data on microfluidic production with varying lipids. Artificial intelligence offers promise in predicting lipid behaviour during microfluidic production, with the still unexploited potential of streamlining development. Herein we employ machine learning to predict critical quality attributes and process parameters for microfluidic-based liposome production. Validated models predict liposome formation, size, and production parameters, significantly advancing our understanding of lipid behaviour. Extensive model analysis enhanced interpretability and investigated underlying mechanisms, supporting the transition to microfluidic production. Unlocking the potential of machine learning in drug development can accelerate pharmaceutical innovation, making drug delivery systems more adaptable and accessible. (https://doi.org/10.1101/2024.07.01.600773)

## Environment

To ensure reproducibility and ease of setup, the project was developed using Python 3.11.9 within an Anaconda 24.3.0 (https://www.anaconda.com/) environment.

## Dependencies

The project uses the following Python packages and libraries (Complete environment is available upon request on Anaconda):

### Data manipulation and analysis
pandas (v. 2.2.2)
numpy (v. 1.26.4)
rdkit (v. 2024.03.2)

### Machine learning models and tools
xgboost (v. 2.0.3)
scikit-learn (v. 1.4.2)

### Visualization
plotly (v. 5.22.0)
seaborn (v. 0.12.2)
shap (0.45.1)

## License

MIT License

Copyright (c) [2024] [Luciani Group]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
