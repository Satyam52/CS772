# Generated by Django 4.1.1 on 2022-10-22 15:29

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('restapi', '0016_user_bids'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Bids',
            new_name='Bid',
        ),
        migrations.AddField(
            model_name='cycle',
            name='user_id',
            field=models.ForeignKey(default=8, on_delete=django.db.models.deletion.CASCADE, to='restapi.user'),
            preserve_default=False,
        ),
    ]