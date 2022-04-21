import os
import typing

import numpy as np
import scipy as sp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import gpytorch


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False  # False
# print('EXTENDED_EVALUATION', EXTENDED_EVALUATION)
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self, plot_points=False):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        
        # number of subsamples
        # total limit (not used in this submission)
        self.CAP = 12000  
        # regional limits
        self.no_high = 5000 
        self.no_low = 4000  
        
        # number of training rounds
        self.epochs = 40

        # folder to plot points
        self.output_dir = '/results'
        self.plot_points = plot_points


    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        self.model.eval()
        self.likelihood.eval()

        # plot testing points
        if self.plot_points:
            fig = plt.figure(figsize=(10, 10))

            ax = fig.add_subplot(111)
            ax.scatter(x[:, 0], x[:, 1], c='red');
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title('testing points, number ' + str(len(x)))

            plt.show()
            # fig.savefig('test.png')

            # Save figure to pdf
            figure_path = os.path.join(self.output_dir, 'test.pdf')
            fig.savefig(figure_path)


        x = torch.Tensor(x)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))

        gp_mean = np.zeros(x.shape[0], dtype=float)
        gp_std = np.zeros(x.shape[0], dtype=float)

        # invert the scaling again
        gp_mean = self.scaler.inverse_transform(observed_pred.mean.detach().numpy())
        gp_var = self.scaler.scale_**2 * observed_pred.variance.detach().numpy()
        gp_std = np.sqrt(gp_var)

        # Simply predict mean (not yet adapted for asymmetric cost function)
        predictions = gp_mean

        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        np.random.seed(0)
        torch.manual_seed(0)

       # scaling
        self.scaler = StandardScaler()
        y_train_scaled = self.scaler.fit_transform(train_y.reshape(-1, 1)).reshape(-1, )

        # subsampling
        # self.no_samples = len(train_x)
        # ind = np.random.randint(0, high=len(train_x), size=min(self.no_samples,self.CAP))
        # x_train_subsample = train_x[ind, :]
        # y_train_subsample = y_train_scaled[ind]

        # subsampling regions
        train_x_up = train_x[train_x[:,1] > 0.5]
        train_y_up = y_train_scaled[train_x[:,1] > 0.5]

        train_x_upl = train_x_up[train_x_up[:,0] <= 0.5]
        train_y_upl = train_y_up[train_x_up[:,0] <= 0.5]
        train_x_upr = train_x_up[train_x_up[:,0] > 0.5]
        train_y_upr = train_y_up[train_x_up[:,0] > 0.5]

        train_x_down = train_x[train_x[:,1] <= 0.5]
        train_y_down = y_train_scaled[train_x[:,1] <= 0.5]

        no_samples = len(train_x_upl)
        ind = np.random.randint(0, high=len(train_x_upl), size=min(no_samples,self.no_high))
        train_x_upl = train_x_upl[ind, :]
        train_y_upl = train_y_upl[ind]

        no_samples = len(train_x_upr)
        ind = np.random.randint(0, high=len(train_x_upr), size=min(no_samples,self.no_low))
        train_x_upr = train_x_upr[ind, :]
        train_y_upr = train_y_upr[ind]

        train_x_up=np.concatenate([train_x_upl,train_x_upr])
        train_y_up=np.concatenate([train_y_upl,train_y_upr])

        no_samples = len(train_x_down)
        ind = np.random.randint(0, high=len(train_x_down), size=min(no_samples,self.no_high))
        train_x_down = train_x_down[ind, :]
        train_y_down = train_y_down[ind]

        x_train_subsample=np.concatenate([train_x_up,train_x_down])
        y_train_subsample=np.concatenate([train_y_up,train_y_down])

        # plot training points
        if self.plot_points:
            fig = plt.figure(figsize=(10, 10))

            ax = fig.add_subplot(111)
            ax.scatter(x_train_subsample[:, 0], x_train_subsample[:, 1], c='blue');
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title('training points, number ' + str(len(x_train_subsample)))

            plt.show()
            # fig.savefig('test.png')

            # Save figure to pdf
            figure_path = os.path.join(self.output_dir, 'train.pdf')
            fig.savefig(figure_path)


        # transform to torch tensors
        x_train_tt = torch.Tensor(x_train_subsample)
        y_train_tt = torch.Tensor(y_train_subsample)


        # the following is mostly from to the official tutorial https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
        # define likelihood and initialize GP Model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(x_train_tt, y_train_tt, self.likelihood)

        EMLL = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model )

        # set-up training
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model .parameters(), lr=0.1)
        losses = []

        # start training
        for i in range(self.epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = self.model (x_train_tt)

            # Calc loss and backprop gradients
            loss = -EMLL(output, y_train_tt)
            loss.backward()

            # Print state
            if ((i + 1) % 5 == 0):
                print('Iter %d/%d - Loss: %.2f   lengthscale: %.2f   noise: %.2f' % (
                    i + 1, self.epochs, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))

            losses.append(loss.item())

            # Take a step on the otimizer
            optimizer.step()

        pass

# from the tutorial https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
