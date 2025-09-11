import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import os


class PCPEDataExploratoryAnalysis:
    """
    A modular class for comprehensive exploratory data analysis.
    Provides various descriptive analysis methods for financial transaction data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the PCPEDataExploratoryAnalysis class.
        
        Args:
            df: DataFrame containing the data to analyze
        """
        self.df = df.copy()
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist in the dataframe."""
        required_columns = ['NUMERO_BANCO', 'NUMERO_AGENCIA', 'NUMERO_CONTA', 'CPF_CNPJ_TITULAR']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary containing basic dataset information
        """
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }

    def get_missing_data_analysis(self) -> pd.DataFrame:
        """
        Analyze missing data patterns.
        
        Returns:
            DataFrame with missing data statistics
        """
        missing_data = pd.DataFrame({
            'missing_count': self.df.isnull().sum(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'non_missing_count': self.df.notnull().sum()
        })
        return missing_data.sort_values('missing_percentage', ascending=False)

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data quality summary.
        
        Returns:
            Dictionary with data quality metrics
        """
        return {
            'total_records': len(self.df),
            'unique_banks': self.df['NUMERO_BANCO'].nunique(),
            'unique_agencies': self.df['NUMERO_AGENCIA'].nunique(),
            'unique_accounts': self.df['NUMERO_CONTA'].nunique(),
            'unique_individuals': self.df['CPF_CNPJ_TITULAR'].nunique(),
            'duplicate_transactions': self.df.duplicated().sum(),
            'missing_values': self.df.isnull().sum().sum()
        }

    def get_typology_analysis(self, typology_column: str = 'I-d', typology_name: str = None) -> pd.DataFrame:
        """
        Analyze data by specified typology column.
        
        Args:
            typology_column: Name of the column containing typology flags (default: 'I-d')
            typology_name: Custom name for the typology (default: uses column name)
        
        Returns:
            Pivot table with typology analysis
        """
        COL_TYPOLOGY = 'Typology'
        COL_ACCOUNT = 'Accounts'
        COL_TRANSACTIONS = 'Transactions'
        COL_INDIVIDUALS = 'Individuals/Companies'

        # Validate that the typology column exists
        if typology_column not in self.df.columns:
            available_columns = [col for col in self.df.columns if col not in ['NUMERO_BANCO', 'NUMERO_AGENCIA', 'NUMERO_CONTA', 'CPF_CNPJ_TITULAR']]
            raise ValueError(f"Typology column '{typology_column}' not found. Available columns: {available_columns}")

        df_descriptive = self.df.copy()
        df_descriptive[COL_TYPOLOGY] = 'None'
        df_descriptive[COL_ACCOUNT] = (
            df_descriptive['NUMERO_BANCO'].astype(str) + 
            df_descriptive['NUMERO_AGENCIA'].astype(str) + 
            df_descriptive['NUMERO_CONTA'].astype(str)
        )
        df_descriptive[COL_TRANSACTIONS] = range(len(df_descriptive))
            
        df_descriptive = df_descriptive.rename(columns={
            'CPF_CNPJ_TITULAR': COL_INDIVIDUALS
        })

        # Apply typology classification
        if typology_name is None:
            typology_name = typology_column
            
        # Mark records with the specified typology
        df_descriptive.loc[df_descriptive[typology_column].notna(), COL_TYPOLOGY] = typology_name

        pivot_table = df_descriptive.pivot_table(
            index=None, 
            columns=COL_TYPOLOGY, 
            values=[COL_TRANSACTIONS, COL_ACCOUNT, COL_INDIVIDUALS], 
            aggfunc=pd.Series.nunique
        )
        return pivot_table

    def get_bank_analysis(self) -> pd.DataFrame:
        """
        Analyze data distribution by bank.
        
        Returns:
            DataFrame with bank-level statistics
        """
        bank_stats = self.df.groupby('NUMERO_BANCO').agg({
            'NUMERO_AGENCIA': 'nunique',
            'NUMERO_CONTA': 'nunique',
            'CPF_CNPJ_TITULAR': 'nunique'
        }).rename(columns={
            'NUMERO_AGENCIA': 'unique_agencies',
            'NUMERO_CONTA': 'unique_accounts',
            'CPF_CNPJ_TITULAR': 'unique_individuals'
        })
        
        bank_stats['transaction_count'] = self.df['NUMERO_BANCO'].value_counts()
        bank_stats['transaction_percentage'] = (
            bank_stats['transaction_count'] / len(self.df) * 100
        )
        
        return bank_stats.sort_values('transaction_count', ascending=False)

    def get_agency_analysis(self) -> pd.DataFrame:
        """
        Analyze data distribution by agency.
        
        Returns:
            DataFrame with agency-level statistics
        """
        agency_stats = self.df.groupby(['NUMERO_BANCO', 'NUMERO_AGENCIA']).agg({
            'NUMERO_CONTA': 'nunique',
            'CPF_CNPJ_TITULAR': 'nunique'
        }).rename(columns={
            'NUMERO_CONTA': 'unique_accounts',
            'CPF_CNPJ_TITULAR': 'unique_individuals'
        })
        
        agency_stats['transaction_count'] = (
            self.df.groupby(['NUMERO_BANCO', 'NUMERO_AGENCIA']).size()
        )
        
        return agency_stats.sort_values('transaction_count', ascending=False)

    def get_individual_analysis(self) -> pd.DataFrame:
        """
        Analyze data distribution by individuals/companies.
        
        Returns:
            DataFrame with individual-level statistics
        """
        individual_stats = self.df.groupby('CPF_CNPJ_TITULAR').agg({
            'NUMERO_BANCO': 'nunique',
            'NUMERO_AGENCIA': 'nunique',
            'NUMERO_CONTA': 'nunique'
        }).rename(columns={
            'NUMERO_BANCO': 'unique_banks',
            'NUMERO_AGENCIA': 'unique_agencies',
            'NUMERO_CONTA': 'unique_accounts'
        })
        
        individual_stats['transaction_count'] = self.df['CPF_CNPJ_TITULAR'].value_counts()
        
        return individual_stats.sort_values('transaction_count', ascending=False)

    def get_account_analysis(self) -> pd.DataFrame:
        """
        Analyze data distribution by account.
        
        Returns:
            DataFrame with account-level statistics
        """
        account_stats = self.df.groupby(['NUMERO_BANCO', 'NUMERO_AGENCIA', 'NUMERO_CONTA']).agg({
            'CPF_CNPJ_TITULAR': 'nunique'
        }).rename(columns={
            'CPF_CNPJ_TITULAR': 'unique_individuals'
        })
        
        account_stats['transaction_count'] = (
            self.df.groupby(['NUMERO_BANCO', 'NUMERO_AGENCIA', 'NUMERO_CONTA']).size()
        )
        
        return account_stats.sort_values('transaction_count', ascending=False)

    def get_cross_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Perform cross-analysis between different dimensions.
        
        Returns:
            Dictionary containing various cross-analysis results
        """
        results = {}
        
        # Bank vs Agency analysis
        results['bank_agency'] = (
            self.df.groupby(['NUMERO_BANCO', 'NUMERO_AGENCIA'])
            .size()
            .unstack(fill_value=0)
        )
        
        # Bank vs Individual analysis
        results['bank_individual'] = (
            self.df.groupby(['NUMERO_BANCO', 'CPF_CNPJ_TITULAR'])
            .size()
            .unstack(fill_value=0)
        )
        
        # Agency vs Individual analysis
        results['agency_individual'] = (
            self.df.groupby(['NUMERO_AGENCIA', 'CPF_CNPJ_TITULAR'])
            .size()
            .unstack(fill_value=0)
        )
        
        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'basic_info': self.get_basic_info(),
            'data_quality': self.get_data_quality_summary(),
            'missing_data': self.get_missing_data_analysis(),
            'top_banks': self.get_bank_analysis().head(10),
            'top_agencies': self.get_agency_analysis().head(10),
            'top_individuals': self.get_individual_analysis().head(10),
            'top_accounts': self.get_account_analysis().head(10)
        }

    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            String containing the analysis report
        """
        summary = self.get_summary_statistics()
        
        report = f"""
EXPLORATORY DATA ANALYSIS REPORT
{'='*50}

DATASET OVERVIEW:
- Total Records: {summary['basic_info']['shape'][0]:,}
- Total Columns: {summary['basic_info']['shape'][1]}
- Memory Usage: {summary['basic_info']['memory_usage'] / 1024**2:.2f} MB
- Duplicate Rows: {summary['basic_info']['duplicate_rows']:,}

DATA QUALITY:
- Unique Banks: {summary['data_quality']['unique_banks']:,}
- Unique Agencies: {summary['data_quality']['unique_agencies']:,}
- Unique Accounts: {summary['data_quality']['unique_accounts']:,}
- Unique Individuals: {summary['data_quality']['unique_individuals']:,}
- Missing Values: {summary['data_quality']['missing_values']:,}

TOP 5 BANKS BY TRANSACTION COUNT:
{summary['top_banks'].head().to_string()}

TOP 5 AGENCIES BY TRANSACTION COUNT:
{summary['top_agencies'].head().to_string()}

TOP 5 INDIVIDUALS BY TRANSACTION COUNT:
{summary['top_individuals'].head().to_string()}

MISSING DATA ANALYSIS:
{summary['missing_data'].to_string()}
        """
        
        return report


    def get_typology_summary(self, typology_column: str = 'I-d') -> Dict[str, Any]:
        """
        Get summary statistics for a specific typology.
        
        Args:
            typology_column: Name of the typology column to analyze
        
        Returns:
            Dictionary with typology summary statistics
        """
        if typology_column not in self.df.columns:
            raise ValueError(f"Typology column '{typology_column}' not found.")
        
        typology_data = self.df[typology_column]
        
        return {
            'total_records': len(self.df),
            'typology_present': typology_data.notna().sum(),
            'typology_absent': typology_data.isna().sum(),
            'typology_percentage': (typology_data.notna().sum() / len(self.df)) * 100,
            'unique_typology_values': typology_data.nunique(),
            'most_common_typology': typology_data.value_counts().index[0] if typology_data.notna().any() else None,
            'most_common_count': typology_data.value_counts().iloc[0] if typology_data.notna().any() else 0
        }

    # Legacy method for backward compatibility
    def get_results_descriptive(self, typology_column: str = 'I-d', typology_name: str = None) -> pd.DataFrame:
        """
        Legacy method - use get_typology_analysis() instead.
        
        Args:
            typology_column: Name of the column containing typology flags (default: 'I-d')
            typology_name: Custom name for the typology (default: uses column name)
        
        Returns:
            Pivot table with typology analysis
        """
        return self.get_typology_analysis(typology_column, typology_name)

    # ==================== LATEX EXPORT METHODS ====================

    def _format_latex_table(self, df: pd.DataFrame, caption: str = "", label: str = "", 
                           index: bool = True, escape: bool = True, 
                           float_format: str = "%.2f") -> str:
        """
        Helper method to format DataFrame as LaTeX table with common settings.
        
        Args:
            df: DataFrame to convert to LaTeX
            caption: Table caption
            label: Table label for referencing
            index: Whether to include index in the table
            escape: Whether to escape special LaTeX characters
            float_format: Format for floating point numbers
        
        Returns:
            LaTeX formatted string
        """
        latex_string = df.to_latex(
            index=index,
            escape=escape,
            float_format=float_format,
            caption=caption,
            label=label,
            position='H'  # Force table position
        )
        return latex_string

    def export_typology_analysis_latex(self, typology_column: str = 'I-d', 
                                     typology_name: str = None,
                                     caption: str = None, 
                                     label: str = None) -> str:
        """
        Export typology analysis as LaTeX table.
        
        Args:
            typology_column: Name of the typology column to analyze
            typology_name: Custom name for the typology
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        df = self.get_typology_analysis(typology_column, typology_name)
        
        if caption is None:
            caption = f"Typology Analysis - {typology_column}"
        if label is None:
            label = f"tab:typology_{typology_column.lower().replace('-', '_')}"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_bank_analysis_latex(self, top_n: int = 10, 
                                 caption: str = None, 
                                 label: str = None) -> str:
        """
        Export bank analysis as LaTeX table.
        
        Args:
            top_n: Number of top banks to include
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        df = self.get_bank_analysis().head(top_n)
        
        if caption is None:
            caption = f"Top {top_n} Banks by Transaction Count"
        if label is None:
            label = "tab:bank_analysis"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_agency_analysis_latex(self, top_n: int = 10, 
                                   caption: str = None, 
                                   label: str = None) -> str:
        """
        Export agency analysis as LaTeX table.
        
        Args:
            top_n: Number of top agencies to include
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        df = self.get_agency_analysis().head(top_n)
        
        if caption is None:
            caption = f"Top {top_n} Agencies by Transaction Count"
        if label is None:
            label = "tab:agency_analysis"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_individual_analysis_latex(self, top_n: int = 10, 
                                       caption: str = None, 
                                       label: str = None) -> str:
        """
        Export individual analysis as LaTeX table.
        
        Args:
            top_n: Number of top individuals to include
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        df = self.get_individual_analysis().head(top_n)
        
        if caption is None:
            caption = f"Top {top_n} Individuals/Companies by Transaction Count"
        if label is None:
            label = "tab:individual_analysis"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_account_analysis_latex(self, top_n: int = 10, 
                                    caption: str = None, 
                                    label: str = None) -> str:
        """
        Export account analysis as LaTeX table.
        
        Args:
            top_n: Number of top accounts to include
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        df = self.get_account_analysis().head(top_n)
        
        if caption is None:
            caption = f"Top {top_n} Accounts by Transaction Count"
        if label is None:
            label = "tab:account_analysis"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_missing_data_analysis_latex(self, caption: str = None, 
                                         label: str = None) -> str:
        """
        Export missing data analysis as LaTeX table.
        
        Args:
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        df = self.get_missing_data_analysis()
        
        if caption is None:
            caption = "Missing Data Analysis"
        if label is None:
            label = "tab:missing_data"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_data_quality_summary_latex(self, caption: str = None, 
                                        label: str = None) -> str:
        """
        Export data quality summary as LaTeX table.
        
        Args:
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        quality_data = self.get_data_quality_summary()
        df = pd.DataFrame(list(quality_data.items()), columns=['Metric', 'Value'])
        
        if caption is None:
            caption = "Data Quality Summary"
        if label is None:
            label = "tab:data_quality"
        
        return self._format_latex_table(df, caption=caption, label=label, index=False)

    def export_cross_analysis_latex(self, analysis_type: str = 'bank_agency',
                                  caption: str = None, 
                                  label: str = None) -> str:
        """
        Export cross-analysis as LaTeX table.
        
        Args:
            analysis_type: Type of cross-analysis ('bank_agency', 'bank_individual', 'agency_individual')
            caption: Custom caption for the table
            label: Custom label for the table
        
        Returns:
            LaTeX formatted string
        """
        cross_analysis = self.get_cross_analysis()
        
        if analysis_type not in cross_analysis:
            available_types = list(cross_analysis.keys())
            raise ValueError(f"Analysis type '{analysis_type}' not found. Available types: {available_types}")
        
        df = cross_analysis[analysis_type]
        
        if caption is None:
            caption = f"Cross Analysis - {analysis_type.replace('_', ' vs ').title()}"
        if label is None:
            label = f"tab:cross_{analysis_type}"
        
        return self._format_latex_table(df, caption=caption, label=label)

    def export_all_analyses_latex(self, output_dir: str = "latex_tables", 
                                top_n: int = 10,
                                typology_columns: List[str] = None) -> Dict[str, str]:
        """
        Export all analyses as LaTeX tables and save to files.
        
        Args:
            output_dir: Directory to save LaTeX files
            top_n: Number of top items to include in ranking tables
            typology_columns: List of typology columns to analyze (default: all available)
        
        Returns:
            Dictionary mapping analysis names to LaTeX strings
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        latex_tables = {}
        
        # Basic analyses
        latex_tables['data_quality'] = self.export_data_quality_summary_latex()
        latex_tables['missing_data'] = self.export_missing_data_analysis_latex()
        latex_tables['bank_analysis'] = self.export_bank_analysis_latex(top_n=top_n)
        latex_tables['agency_analysis'] = self.export_agency_analysis_latex(top_n=top_n)
        latex_tables['individual_analysis'] = self.export_individual_analysis_latex(top_n=top_n)
        latex_tables['account_analysis'] = self.export_account_analysis_latex(top_n=top_n)
        
        # Cross analyses
        for analysis_type in ['bank_agency', 'bank_individual', 'agency_individual']:
            try:
                latex_tables[f'cross_{analysis_type}'] = self.export_cross_analysis_latex(analysis_type)
            except Exception as e:
                print(f"Warning: Could not generate cross analysis for {analysis_type}: {e}")
        
        # Typology analyses
        if typology_columns is None:
            raise ValueError("typology_columns is required")
        
        for typology_col in typology_columns:
            try:
                latex_tables[f'typology_{typology_col}'] = self.export_typology_analysis_latex(typology_col)
            except Exception as e:
                print(f"Warning: Could not generate typology analysis for {typology_col}: {e}")
        
        # Save to files
        for name, latex_content in latex_tables.items():
            filename = os.path.join(output_dir, f"{name}.tex")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            print(f"Saved LaTeX table: {filename}")
        
        return latex_tables

    def generate_latex_document(self, output_file: str = "analysis_report.tex",
                              title: str = "Exploratory Data Analysis Report",
                              author: str = "Data Analysis Team",
                              top_n: int = 10,
                              typology_columns: List[str] = None) -> str:
        """
        Generate a complete LaTeX document with all analyses.
        
        Args:
            output_file: Output file path
            title: Document title
            author: Document author
            top_n: Number of top items to include in ranking tables
            typology_columns: List of typology columns to analyze
        
        Returns:
            Path to the generated LaTeX file
        """
        latex_tables = self.export_all_analyses_latex(
            output_dir=os.path.dirname(output_file) or ".",
            top_n=top_n,
            typology_columns=typology_columns
        )
        
        # Generate LaTeX document
        latex_doc = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{booktabs}}
\\usepackage{{float}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}

\\title{{{title}}}
\\author{{{author}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Data Quality Summary}}
{latex_tables.get('data_quality', '')}

\\section{{Missing Data Analysis}}
{latex_tables.get('missing_data', '')}

\\section{{Bank Analysis}}
{latex_tables.get('bank_analysis', '')}

\\section{{Agency Analysis}}
{latex_tables.get('agency_analysis', '')}

\\section{{Individual/Company Analysis}}
{latex_tables.get('individual_analysis', '')}

\\section{{Account Analysis}}
{latex_tables.get('account_analysis', '')}

\\section{{Cross Analysis}}
\\subsection{{Bank vs Agency}}
{latex_tables.get('cross_bank_agency', '')}

\\subsection{{Bank vs Individual}}
{latex_tables.get('cross_bank_individual', '')}

\\subsection{{Agency vs Individual}}
{latex_tables.get('cross_agency_individual', '')}

\\section{{Typology Analysis}}
"""
        
        # Add typology analyses
        for name, content in latex_tables.items():
            if name.startswith('typology_'):
                typology_name = name.replace('typology_', '').replace('_', ' ').title()
                latex_doc += f"\\subsection{{{typology_name}}}\n{content}\n"
        
        latex_doc += "\\end{document}"
        
        # Save document
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_doc)
        
        print(f"Generated LaTeX document: {output_file}")
        return output_file
