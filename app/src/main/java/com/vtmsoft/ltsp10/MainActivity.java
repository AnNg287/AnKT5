package com.vtmsoft.ltsp10;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;


import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;


import com.vtmsoft.ltsp10.adapter.ProductAdapter;
import com.vtmsoft.ltsp10.models.Product;
import com.vtmsoft.ltsp10.models.SaleManager;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    ListView lv;
    ProductAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //ActionBar actionBar = getSupportActionBar();
        //actionBar.setTitle("DANH SÁCH");
        //actionBar.setDisplayHomeAsUpEnabled(true);
        //actionBar.setHomeAsUpIndicator(R.drawable.ic_menu);

        lv = (ListView) findViewById(R.id.lvProducts);
        //Khởi tạo các sản phẩm
        SaleManager saleManager = SaleManager.get();
        saleManager.generateProducts();
        //lấy các product từ class saleManager
        ArrayList products = saleManager.getProducts();
        adapter = new ProductAdapter(this, products);//khởi tạo adapter
        lv.setAdapter(adapter);//hiển lên listview
        //long click để xoá
        lv.setOnItemLongClickListener(new ItemLongClickRemove());
        //click để sửa
        lv.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position,
                                    long id) {
                Intent intent = new Intent(MainActivity.this, ProductActivity.class);
                intent.putExtra(ProductActivity.EXTRA_POSITION, position);
                startActivity(intent);
            }
        });
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.btnAdd:
                //Toast.makeText(this, "Item 2 selected", Toast.LENGTH_SHORT).show();
                //Them san pham
                openMyCustom();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private class ItemLongClickRemove implements AdapterView.OnItemLongClickListener {
        @Override
        public boolean onItemLongClick(AdapterView parent, View view, final int position, long id) {
            AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(MainActivity.this);
            alertDialogBuilder.setMessage("Bán có muốn xóa sản phẩm này!");
            alertDialogBuilder.setPositiveButton("Có", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    // xóa sp đang nhấn giữ
                    SaleManager.get().getProducts().remove(position);
                    //cập nhật lại listview
                    adapter.notifyDataSetChanged();

                }
            });
            alertDialogBuilder.setNegativeButton("Không", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    //không làm gì
                }
            });
            alertDialogBuilder.show();
            return true;
        }
    }

    public void openMyCustom() {

        // khởi tạo AlertDialog từ đối tượng Builder. tham số truyền vào ở đây là context.
        //final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        final AlertDialog builder = new AlertDialog.Builder(this).create();

        // khoi tao doi tuong View tu file mydialog.
        final View alert = LayoutInflater.from(this).inflate(R.layout.add_item, null);

        // set layout cho alert dialog
        builder.setView(alert);

        final EditText txtProductName;
        final EditText txtUnit;
        final EditText txtPrice;
        Button btnThem;
        Button btnHuy;

        // tham chieu cac doi tuong co tren giao dien dialog vua duoc set
        txtProductName = alert.findViewById(R.id.txtProductName);
        txtUnit = alert.findViewById(R.id.txtUnit);
        txtPrice = alert.findViewById(R.id.txtPrice);
        btnThem = alert.findViewById(R.id.btnThem);
        btnHuy = alert.findViewById(R.id.btnHuy);

        // tạo dialog và hiển thị
        builder.show();

        // bat su kien click vao nut
        btnThem.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Product product = new Product();
                product.setProductName(txtProductName.getText().toString());
                product.setUnit(txtUnit.getText().toString());
                String s = txtPrice.getText().toString();
                s = s.replace(",", "");
                double price = Double.parseDouble(s);
                product.setPrice(price);
                adapter.add(product);
                adapter.notifyDataSetChanged();
                builder.dismiss();
            }
        });

        btnHuy.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                builder.dismiss();
            }
        });

    }

}
