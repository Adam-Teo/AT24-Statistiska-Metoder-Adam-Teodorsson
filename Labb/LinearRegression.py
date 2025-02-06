# LinearRegression.py
import numpy as np
from scipy import stats

class LinearRegression: 
    def __init__(self, X, Y, features, response):
        self._X = X 
        self.Y = Y 
        self.features = features
        self.response = response  
 
    @property 
    def X(self):
        X = self._X
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    @property
    def n(self):
        return self._X.shape[0]
    
    @property
    def d(self):
        return self._X.shape[1] 
     
    @property 
    def b(self):
        X = self.X 
        Y = self.Y
        return np.linalg.pinv( X.T @ X) @ X.T @ Y
        
    # SSE | Residual Sum of Squares | Sum of Squared Error
    # The closer this is to zero the more accuret the prediction (in theory)
    @property 
    def SSE(self):
        X, Y, b = self.X, self.Y, self.b
        return  sum( np.square( Y - X @ b ) )
    
    # SSR | Explained Sum of Squares | Regression Sum of Squares
    @property
    def SSR(self):
        X, Y_mean, b = self.X, self.Y.mean(), self.b
        return sum( np.square( X @ b - Y_mean ) )
    
    # SST | Syy | Total Sum of Squares
    @property 
    def SST(self):
        Y = self.Y
        Y_mean = Y.mean() 
        return sum( np.square(Y-Y_mean) )
    
    # Picks the confidence level
    # If it's below 0.68 it sets it to R2 
    @property 
    def confidence_level(self):
        R2 = self.R2()  
        confidence_levels = [0.997, 0.95, 0.9, 0.8, 0.68, R2]
        return [ cl for cl in confidence_levels if cl <= R2 ][0]

    def print_all(self):
        all = f"""
G
1.     Features: {self.d}
2.  Sample Size: {self.n}
3.     Variance: {self.var()[0]:.5f}
4.  S.Diviation: {self.std()[0]:.5f}
5. Significance: {self.sig()[0]}
6.    Relevance: {self.R2()[0]:.5f}

VG
1. Individual Significance
{ str().join( f"{row}\n{str()}" for row in self.sig_var().split("\n") ) }2. Pairs of Pearsons
{ str().join( f"{row}\n{str()}" for row in self.pearson().split("\n") ) }3. Confidence Interval 
{ str().join( f"{row}\n{str()}" for row in self.con_int().split("\n") ) }4. Confidence Level: {self.confidence_level}
         """
        print(all)

    # The Method that Calculates The Variance | S^ | sigma^2 
    # On average how much our predicted responses will diviate from the regression line
    def var(self):
        SSE, n, d = self.SSE, self.n, self.d 
        return SSE/(n - d - 1)

    # The Method that Calculated The Standard Deviation | S | sigma
    # Measure the same thing as the Variances but in a more readable but less fair unit
    def std(self):
        var = self.var()
        return np.sqrt(var)
    
    # The Method that reports The Significance of the Regression
    # The closer the pvalue is to zero the less likely it is that the correlation
    # we observe between the features and the respons is coincidental. 
    # We want the p-value to be less than 0.05 aka 5% to confidently reject the H0 hypothesis
    def sig(self): 
        SSR, d, n, var = self.SSR, self.d, self.n, self.var()
        
        sig_statistic = (SSR/d)/var
     
        # Survival Function of the F-Distrubution
        p_significance = stats.f.sf(sig_statistic, d, n-d-1)
        return p_significance
    
    # The method that reports The Relevance of Regression
    # Reports how big of a range our model can reliably predict. 
    # So if our R2 value is 0.90 than we could predict 90% of 
    # all responses within, or relatively close to, the standard deviation
    def R2(self):
        SSR, SST = self.SSR, self.SST
        return SSR/SST


    # Significance of the Variables
    # This reports the significance each feature have on the model
    def sig_var(self):   
        X, b, d, n, std, var, features = self.X, self.b, self.d, self.n, self.std(), self.var(), self.features
    
        # Variance/Covariance Matrix
        c = np.linalg.pinv( (X.T @ X) )*var

        # Significans Statisitca Array
        ssa = [ b[i]/(std * np.sqrt(c[i,i])) for i in range(1, c.shape[1])]
        
        cdf = stats.t.cdf(ssa, n-d-1)
        sf =  stats.t.sf(ssa, n-d-1)
        p = [ 2 * min(cdf[idx], sf[idx]) for idx in range(len(ssa)) ]

        result = str().join( f"{features[idx]:<10}: pvalue = {p[idx][0]}\n" for idx in range(len(p))  )
        return result

    # The Method that calculates the Pearson number between all pairs of parameters
    # Reports how muc correlation exists between each pair of features
    # the closer these values are to zero the less correlation exists between the pair
    def pearson(self):
        X, features = self.X, self.features
        
        result = list()
        
        X = X[:,1:]
        for idx in range(len(features)):
            for idy in range(idx):
                if idy == idx:
                    continue 
                p = stats.pearsonr(X.T[idx], X.T[idy])    
                result.append(f"{features[idx]:<9} VS {features[idy]:<9} : {p[0]:.10f}\n")
        
        return str().join(result[::-1])
    
    # The method that calculates the Confidence Interval
    def con_int(self):
        X,  b, n, d, var, std, features = self.X, self.b, self.n, self.d, self.var(), self.std(), self.features
      
        a = 1-self.confidence_level 
        df = n-d-1
        results = list()

        # Variance/Covariance Matrix
        c = np.linalg.pinv( (X.T @ X) )*var

        for i in range(1,d+1):           
            ci = (b[i], stats.t.ppf(a/2, df) * std * np.sqrt(c[i][i]))

            # Returns the result in the order of low to high
            low_to_high = min((ci[0][0]-ci[1][0]), (ci[0][0]+ci[1][0])),max((ci[0][0]-ci[1][0]),(ci[0][0]+ci[1][0]))
            result = f"{features[i-1]}: {ci[0][0]:.5f} ± {abs(ci[1][0]):.5f} | interval:[{low_to_high[0]:.5f} <> {low_to_high[1]:.5f}]\n"
            results.append(result)
        
        return str().join(results)   
    
 