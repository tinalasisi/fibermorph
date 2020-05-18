

def trim_outliers(df, isdf=True, p1=float(0.1), p2=float(0.9)):
    """
    Function to generate a filtered panda dataframe without extreme values.

    :param df_in:       input dataframe (should be Pandas Dataframe, not tested with numpy, but that might work as well)

    :param col_name:    a string with the column name containing the column where you'll be filtering outliers

    :param p1:          float value of the percentile for the bottom cut-off of filtering (optional, defaults at 0.25)

    :param p2:          float value of the percentile for the top cut-off of filtering (optional, defaults at 0.75)

    :return:            returns a filtered Pandas dataframe containing only the values between the chosen percentile
    cut-offs

    """

    if isdf:
        x_val = df.values
        print(x_val)
        x = x_val
        print(x)
        # x = x_val.reshape(-1, 1)
        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x)
        df_in = pd.DataFrame(x_scaled)
        print(df_in)
        q1 = pd.Series(df_in.quantile(p1))
        q3 = pd.Series(df_in.quantile(p2))
        scaled_df_out = df_in.clip(q1, q3, axis=1)
        print(scaled_df_out)
        array_out = scaler.inverse_transform(scaled_df_out)
        print(array_out)
        df_out = pd.DataFrame(array_out)

        return df_out

    else:
        x_val = df.values
        x = x_val.reshape(-1, 1)
        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x)
        df_in = pd.DataFrame(x_scaled)

        q1 = df_in.quantile(p1)
        q3 = df_in.quantile(p2)
        scaled_df_out = df_in.clip(q1, q3, axis=1)
        array_out = scaler.inverse_transform(scaled_df_out)
        df_out = pd.DataFrame(array_out)

        return df_out
