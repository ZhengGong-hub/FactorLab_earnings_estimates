import pandas as pd

# internal imports
from ml_utils.ml_framework import MLFramework

# Example usage:
if __name__ == "__main__":
    # Assuming df is your DataFrame
    df = pd.read_parquet('output_data/cleaned_data.parquet')
    print(df.columns.tolist())

    # Prepare the data
    feature_cols = ['x_60MAVGTTMROA', 'x_60MAVGTTMROE', 'x_BVEV', 'x_FCFP', 'x_IO_TO', 'x_ROEStab', 'x_SEV', 'x_LogUnadjPrice', 'x_AssetTurn_2', 'x_ChgATO_2', 'x_IndRel_AccrualRatioBS', 'x_IndRel_AccrualRatioCF', 'x_IndRel_AdjAccruals', 'x_IndRel_DebtChg1Y', 'x_IndRel_DepToCapex', 'x_IndRel_EPSEstDispFY1C', 'x_IndRel_EPSSurpC', 'x_IndRel_EPSToSalesChg1Y', 'x_IndRel_FwdEPC', 'x_IndRel_FwdFCFPC', 'x_IndRel_InvToAst', 'x_IndRel_LTDA', 'x_IndRel_MaxRetPayoff', 'x_IndRel_PM5D', 'x_IndRel_PM6M', 'x_IndRel_RONA', 'x_IndRel_SUEC', 'x_IndRel_SolvencyRatio', 'x_OCFAst_2', 'x_REToAst_2', 'x_ROA_2', 'x_RelPrStr_12M', 'x_10DMACD', 'x_24MResRtnVar', 'x_4To52WPrcOsc', 'x_52WSlope', 'x_90DCV', 'x_Alpha12M6MPChg', 'x_Alpha18M6MPChg', 'x_Alpha36M6MPChg', 'x_Alpha60M', 'x_AnnVol12M', 'x_AnnVol1M', 'x_AstAdjChg3YEPS', 'x_BP', 'x_CashAst', 'x_Chg3YOPM', 'x_ETO', 'x_HL1M', 'x_HL52W', 'x_PAdjChg1YEPS', 'x_PAdjChg3YEPS', 'x_PAdjChg3YSales', 'x_PM9M', 'x_PTIToNOA', 'x_PrcTo260DL', 'x_PrcTo52WH', 'x_STO', 'x_SalesToEPSChg', 'x_SalesToInvCap', 'x_ShortIntRatio', 'x_52WVPT20DLag', 'x_5DVolSig', 'x_6MAvgChg1MRecC', 'x_CQDOHtoSales', 'x_CQDRItoSales', 'x_CQPTTMCFtoSales', 'x_CVVolPrc20D', 'x_CVVolPrc30D', 'x_CVVolPrc60D', 'x_Chg4QSalesTrend', 'x_ConsQPosChgTTMEPS', 'x_IO_BreadthStabBY', 'x_IO_BreadthStabHD', 'x_IO_ChgofIO_AM', 'x_IO_ChgofIO_HF', 'x_IO_ChgofNumBY', 'x_IO_Concentration', 'x_IO_Foreign', 'x_IO_InvDuration', 'x_IO_Level', 'x_IO_Level_AM', 'x_IO_StabofChgHD', 'x_LogAssets', 'x_PA52WL20DLag', 'x_Rev3MFY2C', 'x_RevMagFY1C', 'x_STO_6M', 'x_SharpeRatio', 'x_AccrualRatioCF', 'x_AdjAccruals', 'x_AdjAstAdjChg1YFCF', 'x_AdjChgEPStoSales', 'x_AdjEBITDAEV', 'x_Beta60M', 'x_CFIC', 'x_CapExToAst', 'x_CashAdjEV', 'x_CashBurn', 'x_CashEV', 'x_CashP', 'x_DebtChg1Y', 'x_DepToCapex', 'x_DivP', 'x_DivToCF', 'x_EPSStab', 'x_EPSToOCFChg', 'x_FL', 'x_LTDA', 'x_PAdjChg1YOCF', 'x_PAdjChg1YSales', 'x_PM6M', 'x_RONA', 'x_RecTurn', 'x_TobinQ', 'x_YoYChgDA', 'x_IndRel_AstP', 'x_IndRel_CapExToAst', 'x_IndRel_CashEV', 'x_IndRel_CashP', 'x_IndRel_DivP', 'x_IndRel_GPMargin', 'x_IndRel_NCAP', 'x_IndRel_OEA', 'x_IndRel_OEP', 'x_IndRel_PAdjChg1YSales', 'x_IndRel_PM9M', 'x_IndRel_PTMargin', 'x_IndRel_RecTurn', 'x_IndRel_SEV', 'x_IndRel_SP', 'x_IndRel_ShortIntTrdVol', 'x_IndRel_TobinQ', 'x_6MChgTgtPrc', 'x_6MChgTgtPrcGap', 'x_6MChgTgtPrcGapEMA', 'x_AdjEBITP', 'x_AdjSTO_6M', 'x_AstGrwth', 'x_BuyToSellRecLess3MEMA', 'x_Chg1YAmihud', 'x_Chg1YLTDA', 'x_Chg1YROA', 'x_Chg1YTurnover', 'x_ChgSalestoChgEPS', 'x_EBITEstDisp', 'x_IO_NAT', 'x_MVEToTL', 'x_RevEstDisp', 'x_12QTrendTTMEPS', 'x_14DayRSI', 'x_5DMoneyFlowVol', 'x_AdjEPSNumRevFY1C', 'x_AdjEPSNumRevFY2C', 'x_AdjFwdEPC', 'x_AdjFwdFCFPC', 'x_AdjRev3MFY1C', 'x_AdjRevMagC', 'x_ConsQPosChginTTMCF', 'x_EPSEstDispFY1C', 'x_EPSEstDispFY2C', 'x_EPSNumRevFY1C', 'x_PM5D', 'x_PSlopeSERR_26W', 'x_SUEC', 'x_AstAdjChg1YCF', 'x_AstAdjChg1YOCF', 'x_CFEV', 'x_CFP', 'x_CapAcqRatio', 'x_CashCycle', 'x_Chg1YOCF', 'x_Chg1YOPM', 'x_ChgSalesMargin', 'x_EBITDAEV', 'x_FCFEqt', 'x_IndRel_CFROIC', 'x_IndRel_DA', 'x_IndRel_LTDE', 'x_IndRel_ROA', 'x_IndRel_ROE', 'x_IndRel_ROIC', 'x_InvToAsset', 'x_PM1M', 'x_ROE', 'x_ShareChg', 'x_UnexpectedRecChg', 'x_WCAccruals', 'x_IndRel_AssetTurn', 'x_IndRel_BP', 'x_IndRel_CFEV', 'x_IndRel_CFP', 'x_IndRel_CashAst', 'x_IndRel_CashCycle', 'x_IndRel_Chg1YOCF', 'x_IndRel_Chg1YOPM', 'x_IndRel_ChgSalesMargin', 'x_IndRel_EBITDAEV', 'x_IndRel_EBITDAP', 'x_IndRel_EP', 'x_IndRel_FCFEV', 'x_IndRel_FCFEqt', 'x_IndRel_FCFP', 'x_IndRel_InvToAsset', 'x_IndRel_OCFEV', 'x_IndRel_OCFP', 'x_IndRel_PAdjChg1YEPS', 'x_IndRel_PM12M1M', 'x_IndRel_PM1M', 'x_IndRel_PTIP', 'x_IndRel_ShareChg', 'x_IndRel_TotalAccruals', 'x_IndRel_WCAccruals', 'x_130DMinRtn', 'x_AccrualRatioBS', 'x_Adj50DVolSig', 'x_CFAst', 'x_EBITMargin', 'x_GPMargin', 'x_InvToAst', 'x_InvToAstChg1Y', 'x_MaxRetPayoff', 'x_RskAdjRS', 'x_SalesAcc', 'x_StdErr180D']  # Example features
    target_col = 'y_EPSNormalized_surprise'  # Replace with your target column
        
    # Initialize the framework
    ml = MLFramework(df, output_dir=f'output_data/ml_run_{target_col}')

    ml.prepare_data(feature_cols, target_col)
    
    # Train and evaluate models
    scores = ml.train_models(cv=2)
    
    # Train the best model
    ml.train_best_model()
    
    # Evaluate the best model
    metrics = ml.evaluate_model()
    
    # Make predictions
    predictions = ml.predict(ml.X_test)
