#for presentation
library(tidyverse)

#Generate training data
x <- seq(from=-3, to=3, by=0.1)
y <- round(100*(cos(2*x) + sin(x)))
df <- data.frame(y, x)

#Plot the base series
ggplot(df, aes(y=y, x=x)) +
  geom_point() +
  theme_bw()



#Plot a simple linear regression
ggplot(df, aes(y=y, x=x)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE, formula = y ~ poly(x,55)) +
  theme_bw()


#Showing the degrees of poly in a wider fashion
df$x2 <- df$x^2
df$x3 <- df$x^3
lm_model <- lm(y ~ x + x2 + x3, df)
df$new_X <- predict(lm_model, df)

ggplot(df, aes(x = x, y = new_X)) + 
  geom_point() +
  theme_bw()

ggplot(df, aes(x = new_X, y = y)) + 
  geom_point() +
  theme_bw() +
  geom_smooth(method='lm', se=FALSE)
  
