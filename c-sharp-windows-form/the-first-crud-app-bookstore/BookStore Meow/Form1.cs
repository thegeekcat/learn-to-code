
using System.Data;
using System.Data.SqlClient;

namespace BookStore_Meow
{
    public partial class Form1 : Form
    {
        // Add DB connection info
        string strConnect = "Server = <myAzureDBserver>.database.windows.net; database = BookStoreMeow; uid = meow94; pwd = qwer1234!@#$";

        public Form1()
        {
            InitializeComponent();
        }

        // 'Search' Button
        private void btnSearch_Click(object sender, EventArgs e)
        {

            // Make a connection to DB server
            SqlConnection DBConn = new SqlConnection();
            DBConn.ConnectionString = strConnect;
            DBConn.Open();

            if (DBConn.State == System.Data.ConnectionState.Open)
            {
                //MessageBox.Show("Connected!");
            }


            // Add a 'SELECT' Query
            string strQuery = "SELECT * FROM member";
            if (txtboxSearchName.Text.Length > 0)
            {
                strQuery = string.Format("SELECT * FROM member WHERE Names = '{0}'", txtboxSearchName.Text);
            }

            SqlCommand cmd = new SqlCommand();
            cmd.Connection = DBConn;


            // Degging
            //cmd.CommandText = strQuery
            //SqlDataReader dr = cmd.ExecuteReader();


            // Open a table in the Grid view
            SqlDataAdapter adapter = new SqlDataAdapter(strQuery, DBConn);

            DataSet ds = new DataSet();
            adapter.Fill(ds);

            dataGridView1.DataSource = ds.Tables[0];


            DBConn.Close();
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }


        private void txtboxSearchName_TextChanged(object sender, EventArgs e)
        {

        }

        private void dataGridView1_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void dataGridView1_CellContentDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            txtboxSearchName.Text = dataGridView1.Rows[e.RowIndex].Cells[1].Value.ToString();
        }

        private void textBox4_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {

        }

        private void btnAdd_Click(object sender, EventArgs e)
        {
            // Add a query to insert a new row
            string strQuery = string.Format("INSERT INTO member(Names, Addr, Mobile, Email) VALUES ('{0}', '{1}', '{2}', '{3}')",
                                            txtboxName.Text,
                                            txtboxAddr.Text,
                                            txtboxMobile.Text,
                                            txtboxEmail.Text);

            // Make a SQL connection
            SqlConnection DBConn = new SqlConnection();
            DBConn.ConnectionString = strConnect;
            DBConn.Open();

            // Pass the SQL query to server
            SqlCommand cmd = new SqlCommand();
            cmd.Connection = DBConn;
            cmd.CommandText = strQuery;
            SqlDataReader rdr = cmd.ExecuteReader();


        }

        private void btnDel_Click(object sender, EventArgs e)
        {
            // Add a query to delete a row
            string strQuery = string.Format("DELETE FROM member WHERE Names = '{0}'", txtboxDel.Text);


            // Make a SQL Connection
            SqlConnection DBConn = new SqlConnection();
            DBConn.ConnectionString = strConnect;
            DBConn.Open();

            // Pass the SQL Query to server
            SqlCommand cmd = new SqlCommand();
            cmd.Connection = DBConn;
            cmd.CommandText = strQuery;
            SqlDataReader rdr = cmd.ExecuteReader();

        }
    }
}