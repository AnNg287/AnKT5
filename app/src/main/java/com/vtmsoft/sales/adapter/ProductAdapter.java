package com.vtmsoft.sales.adapter;

import android.app.Activity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import com.vtmsoft.sales.R;
import com.vtmsoft.sales.models.Product;

import java.text.DecimalFormat;
import java.util.ArrayList;

public class ProductAdapter  extends ArrayAdapter {
    Activity activity;//activity chứa listview
    public ProductAdapter(Activity activity, ArrayList products) {
        super(activity, 0, products);
        this.activity = activity;
    }

    //hàm hiện thị từng item lên listview
    public View getView(int position, View convertView, ViewGroup viewGroup)
    {
        //position là vị tri của mỗi item. nó sẽ chạy hết mảng
        //lấy layout cho từng item
        if (convertView == null)
        {
            LayoutInflater inflater = activity.getLayoutInflater();
            convertView = inflater.inflate(R.layout.listitem_product,null);
        }
        //lấy các textview trong mỗi view
        TextView tvProductName = (TextView)convertView
                .findViewById(R.id.tvProductName);
        TextView tvUnit = (TextView)convertView.findViewById(R.id.tvUnit);
        TextView tvPrice = (TextView)convertView.findViewById(R.id.tvPrice);
        //hiển thị dư liệu lên từng item của listview ở vị trí position
        Product p = (Product) getItem(position);
        tvProductName.setText(p.getProductName());
        tvUnit.setText(p.getUnit());
        String s = (new DecimalFormat("#,###.##")).format(p.getPrice());
        tvPrice.setText(s);
        return convertView;//trả về 1 view khi đã thiết đặt xong
    }
}
