namespace BookStore_Meow
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            dataGridView1 = new DataGridView();
            btnSearch = new Button();
            label1 = new Label();
            txtboxSearchName = new TextBox();
            btnAdd = new Button();
            txtboxName = new TextBox();
            txtboxDel = new TextBox();
            txtboxMobile = new TextBox();
            txtboxAddr = new TextBox();
            btnDel = new Button();
            txtboxEmail = new TextBox();
            ((System.ComponentModel.ISupportInitialize)dataGridView1).BeginInit();
            SuspendLayout();
            // 
            // dataGridView1
            // 
            dataGridView1.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridView1.Location = new Point(12, 28);
            dataGridView1.Name = "dataGridView1";
            dataGridView1.RowTemplate.Height = 25;
            dataGridView1.Size = new Size(603, 341);
            dataGridView1.TabIndex = 0;
            dataGridView1.CellContentDoubleClick += dataGridView1_CellContentDoubleClick;
            // 
            // btnSearch
            // 
            btnSearch.Location = new Point(621, 54);
            btnSearch.Name = "btnSearch";
            btnSearch.Size = new Size(145, 23);
            btnSearch.TabIndex = 1;
            btnSearch.Text = "Search";
            btnSearch.UseVisualStyleBackColor = true;
            btnSearch.Click += btnSearch_Click;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(621, 28);
            label1.Name = "label1";
            label1.Size = new Size(39, 15);
            label1.TabIndex = 2;
            label1.Text = "Name";
            label1.Click += label1_Click;
            // 
            // txtboxSearchName
            // 
            txtboxSearchName.Location = new Point(666, 25);
            txtboxSearchName.Name = "txtboxSearchName";
            txtboxSearchName.Size = new Size(100, 23);
            txtboxSearchName.TabIndex = 3;
            txtboxSearchName.TextChanged += txtboxSearchName_TextChanged;
            // 
            // btnAdd
            // 
            btnAdd.BackColor = Color.PaleTurquoise;
            btnAdd.ForeColor = Color.Black;
            btnAdd.Location = new Point(540, 389);
            btnAdd.Name = "btnAdd";
            btnAdd.Size = new Size(75, 23);
            btnAdd.TabIndex = 4;
            btnAdd.Text = "Add";
            btnAdd.UseVisualStyleBackColor = false;
            btnAdd.Click += btnAdd_Click;
            // 
            // txtboxName
            // 
            txtboxName.Location = new Point(12, 389);
            txtboxName.Name = "txtboxName";
            txtboxName.Size = new Size(122, 23);
            txtboxName.TabIndex = 5;
            txtboxName.TextAlign = HorizontalAlignment.Center;
            // 
            // txtboxDel
            // 
            txtboxDel.Location = new Point(12, 429);
            txtboxDel.Name = "txtboxDel";
            txtboxDel.Size = new Size(506, 23);
            txtboxDel.TabIndex = 6;
            // 
            // txtboxMobile
            // 
            txtboxMobile.Location = new Point(268, 389);
            txtboxMobile.Name = "txtboxMobile";
            txtboxMobile.Size = new Size(122, 23);
            txtboxMobile.TabIndex = 7;
            txtboxMobile.TextAlign = HorizontalAlignment.Center;
            txtboxMobile.TextChanged += textBox3_TextChanged;
            // 
            // txtboxAddr
            // 
            txtboxAddr.Location = new Point(140, 389);
            txtboxAddr.Name = "txtboxAddr";
            txtboxAddr.Size = new Size(122, 23);
            txtboxAddr.TabIndex = 8;
            txtboxAddr.TextAlign = HorizontalAlignment.Center;
            txtboxAddr.TextChanged += textBox4_TextChanged;
            // 
            // btnDel
            // 
            btnDel.BackColor = Color.LightCoral;
            btnDel.Location = new Point(540, 429);
            btnDel.Name = "btnDel";
            btnDel.Size = new Size(75, 23);
            btnDel.TabIndex = 9;
            btnDel.Text = "Delete";
            btnDel.UseVisualStyleBackColor = false;
            btnDel.Click += btnDel_Click;
            // 
            // txtboxEmail
            // 
            txtboxEmail.Location = new Point(396, 389);
            txtboxEmail.Name = "txtboxEmail";
            txtboxEmail.Size = new Size(122, 23);
            txtboxEmail.TabIndex = 10;
            txtboxEmail.TextAlign = HorizontalAlignment.Center;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(778, 489);
            Controls.Add(txtboxEmail);
            Controls.Add(btnDel);
            Controls.Add(txtboxAddr);
            Controls.Add(txtboxMobile);
            Controls.Add(txtboxDel);
            Controls.Add(txtboxName);
            Controls.Add(btnAdd);
            Controls.Add(txtboxSearchName);
            Controls.Add(label1);
            Controls.Add(btnSearch);
            Controls.Add(dataGridView1);
            Name = "Form1";
            Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)dataGridView1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private DataGridView dataGridView1;
        private Button btnSearch;
        private Label label1;
        private TextBox txtboxSearchName;
        private Button btnAdd;
        private TextBox txtboxName;
        private TextBox txtboxDel;
        private TextBox txtboxMobile;
        private TextBox txtboxAddr;
        private Button btnDel;
        private TextBox txtboxEmail;
    }
}