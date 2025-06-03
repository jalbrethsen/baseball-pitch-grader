# **Pitch-Adjusted Batting Average**

## **Project Overview**

This project aims to provide a more nuanced understanding of batter performance by adjusting traditional batting average (BA) based on the inherent "hittability" of the pitches a batter faces. We achieve this by first training a Multi-Layer Perceptron (MLP) neural network to predict pitch outcomes (walk, strike, in-play) and then using the ratio of predicted "in-play" to predicted "strike" probability as a proxy for pitch hittability. Subsequently, a linear regression model is employed to statistically adjust each batter's observed batting average, accounting for the average difficulty of the pitches they encountered.

The core idea is to normalize a batter's performance: a batter who achieves a certain batting average while facing consistently difficult pitches should be rated higher than a batter achieving the same average against easier pitches.

We show the predicted strike probabilities correlates well to a pitcher's strikeout ability implying we can evaluate pitchers swing-and-miss ability based solely on statcast data. Additionally, we use the hittability score to quantify pitch quality impact on a specific batting statistic like batting average and adjust the statistic to normalize for pitch difficulty. We look at some of the higher adjustment levels and find that it can be used to identify flukey seasons and see how a player's pitch difficulty is impacted by the team around him.

## **Features**

* **Pitch Outcome Classification (MLP):** Trains a neural network to predict whether a pitch will result in a walk, strike, or ball in play.  
* **Hittability Score Generation:** Derives a "hittability score" for each pitch based on the MLP's predicted probability of that pitch resulting in a ball in play divided by the probability of the pitch resulting in a strike.  
* **Batting Average Adjustment:** Implements a regression-based method to adjust a batter's observed batting average, normalizing it for the average hittability of pitches faced.  
* **Control for Batter Skill:** Incorporates career batting average and batter handedness as control variables in the adjustment model to isolate the effect of pitch difficulty.  
* **Comprehensive Statcast Data Usage:** Utilizes a large dataset of Statcast pitch-by-pitch data from 2015 to 2024\.

## **Data**

The project uses detailed pitch-by-pitch data from **MLB Statcast** spanning the **2015 to 2024 seasons**.

**Data Filtering:**

* Only data for batters who have seen **more than 5000 pitches** are included to ensure sufficient sample size and reliable statistics.

**Key Data Points Used:**

* **Pitch-level data:** Pitch type, velocity, movement (horizontal/vertical break), pitch location (plate\_x, plate\_z), count (balls/strikes), previous pitch state (speed,location).  
* **Batter-level data:** Batter ID, handedness, career batting average.  
* **Outcome data:** Binary outcome for each pitch (walk, strike, in-play) for MLP training, and hit/no-hit outcome for batting average calculation.

## **Methodology**

### **1\. Hittability Score Generation (MLP)**

A **Multi-Layer Perceptron (MLP)** neural network is trained to classify the outcome of each pitch.

* **Input Features:** Raw pitch characteristics from Statcast (e.g., release\_speed, release_spin_rate, pfx\_x, pfx\_z, plate\_x, plate\_z, pitch\_type (one-hot encoded), balls, strikes, etc.).  
* **Output Classes:** The MLP is trained to predict one of three discrete outcomes for each pitch:  
  * Walk  
  * Strike  
  * In-Play (any ball put into play, regardless of whether it was a hit or an out)  
* **Hittability Score:** The **predicted probability of a pitch resulting in "In-Play"** (from the MLP's softmax output layer) is used as the hittability\_score. A higher probability indicates a more hittable pitch (i.e., a pitch that is more likely to be put into play by the batter).

### **2\. Batting Average Adjustment (Linear Regression)**

Once each pitch has a hittability\_score, we proceed with adjusting batter batting averages.

* **Aggregation:** For each batter, the following are calculated over the analysis period (e.g., a single season):  
  * **Observed Batting Average (observed\_ba):** Total hits / Total at-bats.  
  * **Average Hittability Faced (avg\_hittability\_faced):** The mean hittability\_score of all pitches faced by that batter.  
* **League Average Hittability:** The overall mean hittability\_score across all pitches in the dataset serves as the baseline for comparison.  
* **Regression Model:** A **linear regression model** is constructed with the following structure:Observed BA=β0​+β1​×Avg Hittability Faced+β2​×Batter Career BA+β3​×Batter Handedness (Dummy)+Error  
  * **Dependent Variable:** Observed BA for each batter.  
  * **Independent Variables:**  
    * Avg Hittability Faced: The primary variable of interest, quantifying the average difficulty of pitches seen.  
    * Batter Career BA: A control variable to account for the batter's inherent skill level.  
    * Batter Handedness: A control variable (e.g., a dummy variable for 'Right-handed') to account for platoon splits.  
* **Adjustment Formula:** The adjusted\_ba for each batter is calculated using the regression coefficient (β1​) for Avg Hittability Faced:Adjusted BA=Observed BA−(β1​×(Avg Hittability Faced−League Avg Hittability))  
  This formula effectively "removes" the estimated impact of facing pitches that were easier or harder than the league average, providing a normalized batting average that reflects the batter's performance independent of pitch difficulty.

## **Technologies Used**

* **Python:** Primary programming language.  
* **Pandas:** For data manipulation and aggregation.  
* **NumPy:** For numerical operations.  
* **Scikit-learn (or TensorFlow/Keras/PyTorch):** For building and training the MLP model. (The provided Python code uses a placeholder; your actual implementation would use one of these ML libraries).  
* **Statsmodels:** For linear regression analysis and statistical reporting.  
* **Matplotlib/Seaborn (Optional):** For data visualization.

## **Setup and Installation**

1. **Clone the Repository:**  
   git clone \<your-repo-url\>  
   cd pitch-difficulty-adjusted-ba

2. **Create a Virtual Environment (Recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows: \`venv\\Scripts\\activate\`

3. **Install Dependencies:**  
   pip install -r requirements.txt


## **Usage**

1. **Feature Selection:**
   * Run the notebook feature_selection.ipynb to see statcast feature relevance to outcomes.
1. **Prepare Your Data:**  
   * Run the notebook prepare_input.ipynb, here we use pybaseball to download the Statcast data from 2015 to 2024 and preprocess it (e.g., handling missing values, manipulate data to get previous pitch features, min/max scale to normalize).  
2. **Train the MLP Model:**  
   * Run the notebook pitch_grader.ipynb that trains the MLP to classify pitch outcomes (Walk/Strike/In-Play).   
   * Save your trained MLP model.  
3. **Evaluate Hittability:**  
   * Run the notebook evaluation.ipynb to perform inference on each pitch in your Statcast dataset and use it to normalize batting average with respect to pitch hittability. This notebook will:  
     * Load the pitch data with hittability\_score calculated.  
     * Aggregate data by batter.  
     * Add control variables (career BA, handedness).  
     * Run the linear regression.  
     * Calculate and display the adjusted batting averages.

## **Interpretation of Results**

The output will include a summary of the linear regression model, showing the coefficients for each factor. The most important coefficient will be for avg\_hittability\_faced.

* **Positive Coefficient for avg\_hittability\_faced:** This indicates that facing more hittable pitches (higher average hittability score) is positively correlated with a higher observed batting average. This is expected.  
* **Adjusted Batting Average:**  
  * If a batter's adjusted\_ba is **higher** than their observed\_ba, it suggests they faced tougher-than-average pitches, and their raw average was deflated by pitch difficulty. Their true hitting skill might be better than their raw average suggests.  
  * If a batter's adjusted\_ba is **lower** than their observed\_ba, it suggests they faced easier-than-average pitches, and their raw average was inflated by pitch difficulty. Their true hitting skill might be slightly less impressive than their raw average indicates.

This adjusted metric provides a more "fair" comparison of batter performance by attempting to neutralize the advantage or disadvantage imposed by the quality of pitches they were thrown.

## **Future Work**

* **More Granular Control Variables:** Incorporate additional factors like pitcher fatigue, game situation (e.g., leverage index), and park factors into the regression model for even more precise adjustments.  
* **Advanced Adjustment Models:** Explore more sophisticated statistical models (e.g., hierarchical models, Bayesian methods) that can better handle varying sample sizes per batter and complex interactions.  
* **Time-Series Analysis:** Analyze how adjusted batting averages change over time for individual players, identifying trends beyond raw performance.  
* **Pitch Sequence Analysis:** The sequence of pitches in an at-bat should influence the outcome, additionally batter's may be better or worse at handling specific pitch sequences. 
* **Visualization Dashboard:** Create interactive visualizations to explore adjusted batting averages, pitch hittability distributions, and individual batter performance.  
* **Pitcher Hittability Adjustment:** Extend the concept to adjust pitcher performance metrics (e.g., ERA, FIP) based on the average hittability of pitches *they throw*.
