using Accord.Statistics.Distributions.Fitting;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Models.Markov;
using Accord.Statistics.Models.Markov.Learning;
using Accord.Statistics.Models.Markov.Topology;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;

[WebService(Namespace = "http://tempuri.org/")]
[WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
// To allow this Web Service to be called from script, using ASP.NET AJAX, uncomment the following line. 
// [System.Web.Script.Services.ScriptService]

public class Service : System.Web.Services.WebService
{
    public Service()
    {
        //Uncomment the following line if using designed components 
        //InitializeComponent();
    }

    [WebMethod]
    public int Classify(double[][][] trainDataSet, int[] trainLabels, double[][] testData, String[] classes)
    {
        int states = 5;
        int dimensionsOfFeatures = 12;
        int numberOfClasses = classes.Length;
        int iterations = 0;
        double tolerance = 0.01;

        HiddenMarkovClassifier<MultivariateNormalDistribution> hmm = new HiddenMarkovClassifier<MultivariateNormalDistribution>
            (numberOfClasses, new Forward(states), new MultivariateNormalDistribution(dimensionsOfFeatures), classes);

        // Create the learning algorithm for the ensemble classifier
        var teacher = new HiddenMarkovClassifierLearning<MultivariateNormalDistribution>(hmm,
            // Train each model using the selected convergence criteria
            i => new BaumWelchLearning<MultivariateNormalDistribution>(hmm.Models[i])
            {
                Tolerance = tolerance,
                Iterations = iterations,

                FittingOptions = new NormalOptions()
                {
                    Regularization = 1e-5
                }
            }
        );
        teacher.Empirical = true;
        teacher.Rejection = false;
        // Run the learning algorithm
        double error = teacher.Run(trainDataSet, trainLabels);

        int predictedResult = hmm.Compute(testData);
        return predictedResult;
    }
}