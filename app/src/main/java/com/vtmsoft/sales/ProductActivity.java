package com.vtmsoft.sales;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.vtmsoft.sales.models.Product;
import com.vtmsoft.sales.models.SaleManager;

import java.text.DecimalFormat;

public class ProductActivity extends AppCompatActivity {
    public final static String EXTRA_POSITION = "position";
    EditText txtProductName,txtUnit,txtPrice;
    Product product;
    int position;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_product);
        //lấy các control trên layout
        txtProductName = (EditText)findViewById(R.id.txtProductName);
        txtUnit = (EditText)findViewById(R.id.txtUnit);
        txtPrice = (EditText)findViewById(R.id.txtPrice);
        //hiển thị lên màng hình
        Intent it = getIntent();
        position = it.getExtras().getInt(EXTRA_POSITION);
        product = (Product) SaleManager.get().getProducts().get(position);
        txtProductName.setText(product.getProductName());
        txtUnit.setText(product.getUnit());
        String s = (new DecimalFormat("#,###.##")).format(product.getPrice());
        txtPrice.setText(s);
        //thiết đặt sự kiện khi click vào các button
        Button btnOK = (Button)findViewById(R.id.btnOK);
        Button btnCancel = (Button)findViewById(R.id.btnCancel);
        btnOK.setOnClickListener(new OKClickListener());
        btnCancel.setOnClickListener(new CancelClickListener());
    }
    class OKClickListener implements View.OnClickListener
    {
        @Override
        public void onClick(View v) {
            //lấy dữ liệu từ layout để cập nhật lại các sản phẩm trong mảng
            product.setProductName(txtProductName.getText().toString());
            product.setUnit(txtUnit.getText().toString());
            String s = txtPrice.getText().toString();
            s = s.replace(",", "");
            double price = Double.parseDouble(s);
            product.setPrice(price);
            Intent returnIntent = new Intent();
            setResult(Activity.RESULT_OK, returnIntent);
            finish();
        }
    }
    class CancelClickListener implements View.OnClickListener
    {
        @Override
        public void onClick(View v) {
            // không làm gì cả và trở về màng hình trước
            Intent returnIntent = new Intent();
            setResult(Activity.RESULT_CANCELED, returnIntent);
            finish();
        }
    }
}
