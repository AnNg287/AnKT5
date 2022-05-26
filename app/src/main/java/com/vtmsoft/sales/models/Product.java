package com.vtmsoft.sales.models;

public class Product {
    private String ProductName;
    private String Unit;

    private double Price;
    public Product() {
        ProductName = "";
        Unit = "";
        Price = 0;
    }

    public Product(String pname, String unit, double price) {
        ProductName = pname;
        Unit = unit;
        Price = price;
    }

    public String getProductName() {
        return ProductName;
    }

    public void setProductName(String productName) {
        ProductName = productName;
    }

    public String getUnit() {
        return Unit;
    }

    public void setUnit(String unit) {
        Unit = unit;
    }

    public double getPrice() {
        return Price;
    }

    public void setPrice(double price) {
        Price = price;
    }
}
